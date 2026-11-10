package jp.roomfit.capture;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Config;
import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingFailureReason;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * RoomFit 撮影（確認版）。
 *
 * 目的は 2 つ:
 *  1. この端末で ARCore が動くか、深度（通常・生）が取れるかを実際に起動して確かめる。
 *  2. 写真・カメラ姿勢・内部パラメータ・深度を、同じ時刻の組として保存する。
 *
 * 保存する値の意味（座標・単位は docs/座標系と単位.md と揃える）:
 *  - 姿勢: ARCore の世界座標（右手系・Y が上＝重力の反対・単位 m）でのカメラの位置と向き。
 *          カメラ座標は OpenGL 規約（+X 右, +Y 上, -Z 前）。OpenCV 規約ではないので、取り込み側で変換する。
 *  - 内部パラメータ: 保存する画像（CPU 画像・センサーの向きのまま）に対する fx, fy, cx, cy [画素]。
 *  - 深度: uint16・リトルエンディアン・単位 mm。0 は「値なし」。画像と同じ向きで、解像度は低い。
 *  - 生深度の信頼度: uint8（0〜255）。誤差の大きさ [cm] ではない。
 */
public class MainActivity extends Activity implements GLSurfaceView.Renderer {
    private static final String TAG = "RoomFitProbe";
    private static final int MAX_KEYFRAMES = 60;
    /** 軽量版: 深度の計算を止め、ピントを固定し、写真の解像度を下げる。位置の追跡が処理の重さで乱れていないかを確かめるため（2026-10-14）。 */
    private static final boolean LITE = false;
    private static final float KEY_MOVE_M = 0.15f;
    private static final float KEY_ROT_DEG = 12f;

    private GLSurfaceView glView;
    private TextView statusView;
    private TextView errorView;
    private Button recordButton;
    private Button shotButton;

    private final Object lock = new Object();
    private Session session;
    private boolean installRequested;
    private boolean textureSet;
    private boolean depthAuto, depthRaw;
    private String availability = "?";
    private String cameraConfigNote = "";

    private int texId = -1, program, aPos, aUv;
    private final FloatBuffer ndc = floats(new float[]{-1, -1, 1, -1, -1, 1, 1, 1});
    private final FloatBuffer uv = floats(new float[8]);
    private int viewW, viewH;

    private long lastStatusMs, lastLogMs;
    private volatile boolean recording;
    private volatile boolean forceShot;
    private File sessionDir;
    private int saved;
    private Pose lastKeyPose;
    private final JSONArray frameIndex = new JSONArray();

    // 実機での記録（2026-10-12）で、速く動かすと位置が 1m 以上飛ぶ場面と、数秒前の生深度が返る場面があった。
    // そのため、速すぎる場面は保存せず、深度はそれが撮られた時刻の姿勢と組にして保存する。
    private static final float MAX_SPEED_MPS = 0.8f;
    private static final float MAX_TURN_DPS = 90f;
    private static final long MIN_SAVE_INTERVAL_NS = 300_000_000L;
    private static final int HIST = 256;
    private final long[] histTs = new long[HIST];
    private final Pose[] histPose = new Pose[HIST];
    private int histN;
    private Pose prevPose;
    private long prevTs, lastSaveTs, stableSinceTs;
    private volatile boolean tooFast;
    private int skippedFast, jumps;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        FrameLayout root = new FrameLayout(this);
        glView = new GLSurfaceView(this);
        glView.setPreserveEGLContextOnPause(true);
        glView.setEGLContextClientVersion(2);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        root.addView(glView, new FrameLayout.LayoutParams(-1, -1));

        // 状態表示は行数を固定し、文字量でカメラの表示領域が伸び縮みしないようにする
        statusView = new TextView(this);
        statusView.setLines(6);
        statusView.setTextSize(13);
        statusView.setTextColor(Color.WHITE);
        statusView.setBackgroundColor(0xB0000000);
        statusView.setPadding(24, 72, 24, 16);
        statusView.setText("準備しています…");
        root.addView(statusView, new FrameLayout.LayoutParams(-1, -2, Gravity.TOP));

        // エラーは通常の状態表示と分け、次に成功するまで消さない
        errorView = new TextView(this);
        errorView.setTextSize(15);
        errorView.setTextColor(Color.WHITE);
        errorView.setBackgroundColor(0xE0B3261E);
        errorView.setPadding(32, 32, 32, 32);
        errorView.setVisibility(View.GONE);
        FrameLayout.LayoutParams elp = new FrameLayout.LayoutParams(-1, -2, Gravity.CENTER);
        elp.setMargins(32, 0, 32, 0);
        root.addView(errorView, elp);

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setBackgroundColor(0xB0000000);
        bar.setPadding(24, 16, 24, 72);
        recordButton = new Button(this);
        recordButton.setText("記録を始める");
        recordButton.setOnClickListener(v -> toggleRecording());
        shotButton = new Button(this);
        shotButton.setText("この場面を追加");
        shotButton.setEnabled(false);
        shotButton.setOnClickListener(v -> forceShot = true);
        bar.addView(recordButton, new LinearLayout.LayoutParams(0, -2, 1));
        bar.addView(shotButton, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(bar, new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM));
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            return;
        }
        startSession();
        glView.onResume();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        if (results.length == 0 || results[0] != PackageManager.PERMISSION_GRANTED) {
            showError("カメラを使う許可がありません。\n設定 → アプリ → RoomFit 撮影 → 権限 で「カメラ」を許可してから、もう一度開いてください。");
        }
        // 許可された場合は、この後に呼ばれる onResume で開始する
    }

    private void startSession() {
        synchronized (lock) {
            try {
                if (session == null) {
                    availability = String.valueOf(ArCoreApk.getInstance().checkAvailability(this));
                    Log.i(TAG, "{\"event\":\"availability\",\"value\":\"" + availability + "\"}");
                    if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                        installRequested = true;
                        return;
                    }
                    Session s = new Session(this);
                    if (!LITE) chooseCameraConfig(s);
                    else cameraConfigNote = "lite: default camera config";
                    Config c = new Config(s);
                    depthAuto = !LITE && s.isDepthModeSupported(Config.DepthMode.AUTOMATIC);
                    depthRaw = !LITE && s.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY);
                    c.setDepthMode(depthAuto ? Config.DepthMode.AUTOMATIC : depthRaw ? Config.DepthMode.RAW_DEPTH_ONLY : Config.DepthMode.DISABLED);
                    c.setFocusMode(LITE ? Config.FocusMode.FIXED : Config.FocusMode.AUTO);
                    c.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
                    c.setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL);
                    s.configure(c);
                    session = s;
                    textureSet = false;
                    Log.i(TAG, new JSONObject().put("event", "session").put("depthAuto", depthAuto).put("depthRaw", depthRaw)
                            .put("cameraConfig", cameraConfigNote).put("model", Build.MODEL).put("android", Build.VERSION.RELEASE).toString());
                }
                session.resume();
                hideError();
            } catch (UnavailableArcoreNotInstalledException | UnavailableUserDeclinedInstallationException e) {
                fail("ARCore（Google Play Services for AR）が入っていないため、撮影を始められません。Google Play から入れてください。", e);
            } catch (UnavailableApkTooOldException e) {
                fail("ARCore（Google Play Services for AR）が古いため使えません。Google Play で更新してください。", e);
            } catch (UnavailableSdkTooOldException e) {
                fail("このアプリが古く、端末の ARCore と合いません。アプリを更新してください。", e);
            } catch (UnavailableDeviceNotCompatibleException e) {
                fail("この端末は ARCore に対応していません。写真アップロードでの復元は RoomFit の Web 版から利用できます。", e);
            } catch (CameraNotAvailableException e) {
                fail("カメラを使えませんでした。ほかのアプリがカメラを使っていないか確認し、もう一度開いてください。", e);
            } catch (Exception e) {
                fail("撮影の準備に失敗しました。アプリを開き直してください。", e);
            }
        }
    }

    /** CPU 画像（保存する写真）の解像度が高い設定を選ぶ。幅 1920 までで最大のもの。 */
    private void chooseCameraConfig(Session s) throws Exception {
        List<CameraConfig> list = s.getSupportedCameraConfigs(new CameraConfigFilter(s));
        CameraConfig best = null;
        JSONArray all = new JSONArray();
        for (CameraConfig cc : list) {
            Size im = cc.getImageSize();
            all.put(im.getWidth() + "x" + im.getHeight() + "/tex" + cc.getTextureSize().getWidth() + "x" + cc.getTextureSize().getHeight()
                    + "/depthSensor=" + cc.getDepthSensorUsage() + "/fps" + cc.getFpsRange());
            if (im.getWidth() > 1920) continue;
            if (best == null || im.getWidth() * im.getHeight() > best.getImageSize().getWidth() * best.getImageSize().getHeight()) best = cc;
        }
        Log.i(TAG, new JSONObject().put("event", "cameraConfigs").put("list", all).toString());
        if (best != null) {
            s.setCameraConfig(best);
            cameraConfigNote = best.getImageSize().getWidth() + "x" + best.getImageSize().getHeight() + " depthSensor=" + best.getDepthSensorUsage();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // バックグラウンドへ移るときは記録を止めて、それまでの分を確実に書き出す
        if (recording) stopRecording("アプリが背面に移ったため、記録を止めました。");
        glView.onPause();
        synchronized (lock) {
            if (session != null) session.pause();
        }
    }

    @Override
    protected void onDestroy() {
        synchronized (lock) {
            if (session != null) {
                session.close();
                session = null;
            }
        }
        super.onDestroy();
    }

    // ------------------------------------------------------------------ 記録

    private void toggleRecording() {
        if (recording) {
            stopRecording(null);
            return;
        }
        File base = getExternalFilesDir("captures");
        if (base == null) {
            showError("保存先を用意できませんでした。端末の空き容量を確認してください。");
            return;
        }
        sessionDir = new File(base, new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()));
        if (!sessionDir.mkdirs()) {
            showError("保存先を作れませんでした。端末の空き容量を確認してください。");
            return;
        }
        saved = 0;
        lastKeyPose = null;
        prevPose = null;
        histN = 0;
        skippedFast = 0;
        jumps = 0;
        tooFast = false;
        lastSaveTs = 0;
        stableSinceTs = 0;
        while (frameIndex.length() > 0) frameIndex.remove(0);
        recording = true;
        recordButton.setText("記録を終える");
        shotButton.setEnabled(true);
    }

    private void stopRecording(String note) {
        recording = false;
        try {
            JSONObject meta = new JSONObject();
            meta.put("format", "roomfit-capture/1");
            meta.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            meta.put("android", Build.VERSION.RELEASE);
            meta.put("arcoreAvailability", availability);
            meta.put("depthAuto", depthAuto).put("depthRaw", depthRaw);
            meta.put("cameraConfig", cameraConfigNote);
            meta.put("world", "ARCore world: right-handed, +Y up (gravity), meters");
            meta.put("cameraFrame", "OpenGL: +X right, +Y up, -Z forward (NOT OpenCV)");
            meta.put("pose", "camera-to-world. t=[x,y,z] m, q=[x,y,z,w]");
            meta.put("intrinsics", "pixels, for the saved JPEG (sensor orientation, not rotated for display)");
            meta.put("depth", "uint16 little-endian, millimeters, 0=no value, same orientation as the JPEG");
            meta.put("confidence", "uint8 0..255 for raw depth; not an error in cm");
            meta.put("rawPose", "rawT/rawQ = pose nearest to rawTimestampNs (use with rawdepth16). rawPoseDiffMs = how far that pose is in time");
            meta.put("skippedTooFast", skippedFast).put("poseJumps", jumps);
            meta.put("frames", frameIndex);
            write(new File(sessionDir, "session.json"), meta.toString(1).getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            Log.e(TAG, "session.json", e);
            showError("記録のまとめを書き出せませんでした。写真は残っています。");
        }
        final String msg = note;
        runOnUiThread(() -> {
            recordButton.setText("記録を始める");
            shotButton.setEnabled(false);
            if (msg != null) showError(msg + "\n（" + saved + " 枚を保存済み）");
        });
        Log.i(TAG, "{\"event\":\"recordingStopped\",\"saved\":" + saved + ",\"dir\":\"" + sessionDir.getName() + "\"}");
    }

    private static float rotationDeg(Pose a, Pose b) {
        float[] q = a.inverse().compose(b).getRotationQuaternion();
        return (float) Math.toDegrees(2 * Math.acos(Math.min(1.0, Math.abs(q[3]))));
    }

    private static float distance(Pose a, Pose b) {
        float dx = a.tx() - b.tx(), dy = a.ty() - b.ty(), dz = a.tz() - b.tz();
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /** 追跡できているときだけ、前の保存から十分動いた場面を保存する。 */
    private void maybeSave(Frame frame, Camera cam) {
        if (cam.getTrackingState() != TrackingState.TRACKING) {
            forceShot = false; // 追跡が外れている間の姿勢は信用できないので保存しない
            return;
        }
        Pose pose = cam.getPose();
        long ts = frame.getTimestamp();
        // 速さの確認（1 つ前のフレームとの差）。位置が飛んだ直後は、0.5 秒落ち着くまで保存しない
        float speed = 0, turn = 0;
        if (prevPose != null && ts > prevTs) {
            float dt = (ts - prevTs) / 1e9f;
            speed = distance(prevPose, pose) / dt;
            turn = rotationDeg(prevPose, pose) / dt;
            if (distance(prevPose, pose) > 0.3f) {
                jumps++;
                // 実機の記録（2026-10-12）で、位置が 15m 飛んでも画面に何も出ず、使えない記録が保存された。
                // 1 フレームで 30cm 以上動くことは手持ちでは起きないので、追跡の失敗として知らせる。
                if (jumps == 1) showError("位置の追跡が飛びました。この記録は形が崩れている可能性があります。\n「記録を終える」を押して、撮り直してください。\n（壁から 1.5m 以上離れ、模様のある所が画面に入るように、ゆっくり動かすと起きにくくなります）");
            }
        }
        prevPose = pose;
        prevTs = ts;
        histTs[histN % HIST] = ts;
        histPose[histN % HIST] = pose;
        histN++;
        if (speed > MAX_SPEED_MPS || turn > MAX_TURN_DPS) {
            stableSinceTs = ts;
            if (!tooFast) skippedFast++;
            tooFast = true;
            return;
        }
        if (ts - stableSinceTs < 500_000_000L) return;
        tooFast = false;
        boolean moved = lastKeyPose == null || distance(lastKeyPose, pose) > KEY_MOVE_M || rotationDeg(lastKeyPose, pose) > KEY_ROT_DEG;
        if ((!moved && !forceShot) || ts - lastSaveTs < MIN_SAVE_INTERVAL_NS) return;
        forceShot = false;
        lastSaveTs = ts;
        if (saved >= MAX_KEYFRAMES) {
            stopRecording("1 回の記録は " + MAX_KEYFRAMES + " 枚までです。記録を止めました。");
            return;
        }
        String id = String.format(Locale.US, "%03d", saved);
        try {
            JSONObject j = new JSONObject();
            j.put("id", id).put("timestampNs", frame.getTimestamp());
            j.put("tracking", String.valueOf(cam.getTrackingState())).put("failure", String.valueOf(cam.getTrackingFailureReason())).put("lite", LITE);
            float[] t = pose.getTranslation(), q = pose.getRotationQuaternion();
            j.put("t", new JSONArray(new double[]{t[0], t[1], t[2]}));
            j.put("q", new JSONArray(new double[]{q[0], q[1], q[2], q[3]}));
            CameraIntrinsics in = cam.getImageIntrinsics();
            j.put("fx", in.getFocalLength()[0]).put("fy", in.getFocalLength()[1]);
            j.put("cx", in.getPrincipalPoint()[0]).put("cy", in.getPrincipalPoint()[1]);
            j.put("w", in.getImageDimensions()[0]).put("h", in.getImageDimensions()[1]);
            try (Image im = frame.acquireCameraImage()) {
                write(new File(sessionDir, id + ".jpg"), toJpeg(im));
                j.put("imageTimestampNs", im.getTimestamp());
            }
            if (depthAuto) {
                try (Image d = frame.acquireDepthImage16Bits()) {
                    write(new File(sessionDir, id + ".depth16"), plane(d, 2));
                    j.put("depthW", d.getWidth()).put("depthH", d.getHeight()).put("depthTimestampNs", d.getTimestamp());
                } catch (NotYetAvailableException e) {
                    j.put("depthMissing", true);
                }
            }
            if (depthAuto || depthRaw) {
                try (Image d = frame.acquireRawDepthImage16Bits(); Image c = frame.acquireRawDepthConfidenceImage()) {
                    write(new File(sessionDir, id + ".rawdepth16"), plane(d, 2));
                    write(new File(sessionDir, id + ".conf8"), plane(c, 1));
                    j.put("rawW", d.getWidth()).put("rawH", d.getHeight()).put("rawTimestampNs", d.getTimestamp());
                    // 生深度は写真より前の時刻のことがある。その時刻に最も近い姿勢を一緒に残す
                    int best = -1;
                    long bestDiff = Long.MAX_VALUE;
                    for (int k = 0; k < Math.min(histN, HIST); k++) {
                        long diff = Math.abs(histTs[k] - d.getTimestamp());
                        if (diff < bestDiff) {
                            bestDiff = diff;
                            best = k;
                        }
                    }
                    if (best >= 0) {
                        float[] rt = histPose[best].getTranslation(), rq = histPose[best].getRotationQuaternion();
                        j.put("rawT", new JSONArray(new double[]{rt[0], rt[1], rt[2]}));
                        j.put("rawQ", new JSONArray(new double[]{rq[0], rq[1], rq[2], rq[3]}));
                        j.put("rawPoseDiffMs", bestDiff / 1e6);
                    }
                } catch (NotYetAvailableException e) {
                    j.put("rawDepthMissing", true);
                }
            }
            write(new File(sessionDir, id + ".json"), j.toString().getBytes(StandardCharsets.UTF_8));
            frameIndex.put(id);
            lastKeyPose = pose;
            saved++;
        } catch (NotYetAvailableException e) {
            // カメラ画像がまだ届いていないだけ。次のフレームで再度試す
        } catch (Exception e) {
            Log.e(TAG, "save failed", e);
            stopRecording("保存に失敗したため、記録を止めました。端末の空き容量を確認してください。");
        }
    }

    private static void write(File f, byte[] data) throws Exception {
        try (FileOutputStream o = new FileOutputStream(f)) {
            o.write(data);
        }
    }

    /** 行の詰め物（rowStride）を除いて、1 面を連続したバイト列にする。 */
    private static byte[] plane(Image im, int bytesPerPixel) {
        Image.Plane p = im.getPlanes()[0];
        ByteBuffer buf = p.getBuffer();
        int w = im.getWidth(), h = im.getHeight(), rs = p.getRowStride();
        byte[] out = new byte[w * h * bytesPerPixel];
        for (int y = 0; y < h; y++) {
            buf.position(y * rs);
            buf.get(out, y * w * bytesPerPixel, w * bytesPerPixel);
        }
        return out;
    }

    private static byte[] toJpeg(Image im) {
        int w = im.getWidth(), h = im.getHeight();
        byte[] nv21 = new byte[w * h * 3 / 2];
        Image.Plane yP = im.getPlanes()[0], uP = im.getPlanes()[1], vP = im.getPlanes()[2];
        ByteBuffer yb = yP.getBuffer(), ub = uP.getBuffer(), vb = vP.getBuffer();
        for (int r = 0; r < h; r++) {
            yb.position(r * yP.getRowStride());
            yb.get(nv21, r * w, w);
        }
        int ups = uP.getPixelStride(), urs = uP.getRowStride(), vps = vP.getPixelStride(), vrs = vP.getRowStride();
        int o = w * h;
        for (int r = 0; r < h / 2; r++) {
            for (int c = 0; c < w / 2; c++) {
                nv21[o++] = vb.get(r * vrs + c * vps);
                nv21[o++] = ub.get(r * urs + c * ups);
            }
        }
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        new YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(new Rect(0, 0, w, h), 92, bos);
        return bos.toByteArray();
    }

    // ------------------------------------------------------------------ 描画と診断

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        int[] t = new int[1];
        GLES20.glGenTextures(1, t, 0);
        texId = t[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        int vs = shader(GLES20.GL_VERTEX_SHADER, "attribute vec4 aPos; attribute vec2 aUv; varying vec2 vUv; void main(){ gl_Position=aPos; vUv=aUv; }");
        int fs = shader(GLES20.GL_FRAGMENT_SHADER, "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 vUv; uniform samplerExternalOES uTex; void main(){ gl_FragColor=texture2D(uTex,vUv); }");
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vs);
        GLES20.glAttachShader(program, fs);
        GLES20.glLinkProgram(program);
        aPos = GLES20.glGetAttribLocation(program, "aPos");
        aUv = GLES20.glGetAttribLocation(program, "aUv");
        textureSet = false;
    }

    private static int shader(int type, String src) {
        int s = GLES20.glCreateShader(type);
        GLES20.glShaderSource(s, src);
        GLES20.glCompileShader(s);
        return s;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int w, int h) {
        viewW = w;
        viewH = h;
        GLES20.glViewport(0, 0, w, h);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        synchronized (lock) {
            if (session == null || texId < 0) return;
            try {
                if (!textureSet) {
                    session.setCameraTextureName(texId);
                    textureSet = true;
                }
                session.setDisplayGeometry(getWindowManager().getDefaultDisplay().getRotation(), viewW, viewH);
                Frame frame = session.update();
                if (frame.hasDisplayGeometryChanged() || uv.get(0) == 0f && uv.get(7) == 0f) {
                    frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndc, Coordinates2d.TEXTURE_NORMALIZED, uv);
                }
                if (frame.getTimestamp() != 0) drawBackground();
                Camera cam = frame.getCamera();
                if (recording) maybeSave(frame, cam);
                long now = System.currentTimeMillis();
                if (now - lastStatusMs > 400) {
                    lastStatusMs = now;
                    report(frame, cam, now - lastLogMs > 1000);
                    if (now - lastLogMs > 1000) lastLogMs = now;
                }
            } catch (CameraNotAvailableException e) {
                fail("カメラが使えなくなりました。アプリを開き直してください。", e);
            } catch (Exception e) {
                Log.e(TAG, "frame", e);
            }
        }
    }

    private void drawBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId);
        ndc.position(0);
        uv.position(0);
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, ndc);
        GLES20.glVertexAttribPointer(aUv, 2, GLES20.GL_FLOAT, false, 0, uv);
        GLES20.glEnableVertexAttribArray(aPos);
        GLES20.glEnableVertexAttribArray(aUv);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }

    /** 中央の値と、値が入っている画素の割合。 */
    private static int[] depthStats(Image d) {
        Image.Plane p = d.getPlanes()[0];
        ByteBuffer b = p.getBuffer().order(ByteOrder.LITTLE_ENDIAN);
        int w = d.getWidth(), h = d.getHeight(), rs = p.getRowStride();
        int center = b.getShort((h / 2) * rs + (w / 2) * 2) & 0xFFFF;
        int valid = 0, n = 0;
        for (int y = 0; y < h; y += 4) {
            for (int x = 0; x < w; x += 4) {
                n++;
                if ((b.getShort(y * rs + x * 2) & 0xFFFF) != 0) valid++;
            }
        }
        return new int[]{center, n == 0 ? 0 : valid * 100 / n, w, h};
    }

    private void report(Frame frame, Camera cam, boolean log) throws Exception {
        TrackingState ts = cam.getTrackingState();
        TrackingFailureReason why = cam.getTrackingFailureReason();
        JSONObject j = new JSONObject();
        j.put("event", "frame").put("tracking", String.valueOf(ts)).put("failure", String.valueOf(why));
        String depthLine, rawLine;
        if (depthAuto) {
            try (Image d = frame.acquireDepthImage16Bits()) {
                int[] s = depthStats(d);
                depthLine = String.format(Locale.US, "画面中央までの距離: %.2f m（深度 %dx%d、値あり %d%%）", s[0] / 1000f, s[2], s[3], s[1]);
                j.put("depthCenterMm", s[0]).put("depthValidPct", s[1]).put("depthSize", s[2] + "x" + s[3]);
            } catch (NotYetAvailableException e) {
                depthLine = "深度: まだ届いていません（端末をゆっくり左右に動かしてください）";
                j.put("depth", "notYet");
            }
        } else {
            depthLine = "深度: この端末では使えません";
        }
        if (depthAuto || depthRaw) {
            try (Image d = frame.acquireRawDepthImage16Bits(); Image c = frame.acquireRawDepthConfidenceImage()) {
                int[] s = depthStats(d);
                int conf = c.getPlanes()[0].getBuffer().get((c.getHeight() / 2) * c.getPlanes()[0].getRowStride() + c.getWidth() / 2) & 0xFF;
                rawLine = s[0] == 0 ? String.format(Locale.US, "生の深度: 中央は値なし（値あり %d%%）", s[1])
                        : String.format(Locale.US, "生の深度: 中央 %.2f m、信頼度 %d/255（値あり %d%%）", s[0] / 1000f, conf, s[1]);
                j.put("rawCenterMm", s[0]).put("rawValidPct", s[1]).put("rawConf", conf).put("rawSize", s[2] + "x" + s[3]);
            } catch (NotYetAvailableException e) {
                rawLine = "生の深度: まだ届いていません";
                j.put("raw", "notYet");
            }
        } else {
            rawLine = "生の深度: この端末では使えません";
        }
        String trackLine;
        if (ts == TrackingState.TRACKING) {
            trackLine = "位置の追跡: できています";
        } else {
            String advice;
            switch (why) {
                case EXCESSIVE_MOTION: advice = "動きが速すぎます。ゆっくり動かしてください"; break;
                case INSUFFICIENT_LIGHT: advice = "暗すぎます。部屋を明るくしてください"; break;
                case INSUFFICIENT_FEATURES: advice = "模様の少ない面です。家具や床の境目が入るように向けてください"; break;
                case CAMERA_UNAVAILABLE: advice = "ほかのアプリがカメラを使っています"; break;
                case BAD_STATE: advice = "アプリを開き直してください"; break;
                default: advice = "端末をゆっくり左右に動かしてください"; break;
            }
            trackLine = "位置の追跡: 準備中・中断（" + advice + "）";
        }
        if (ts == TrackingState.TRACKING) {
            Pose p = cam.getPose();
            CameraIntrinsics in = cam.getImageIntrinsics();
            j.put("t", new JSONArray(new double[]{p.tx(), p.ty(), p.tz()}));
            j.put("fx", in.getFocalLength()[0]).put("fy", in.getFocalLength()[1]);
            j.put("img", in.getImageDimensions()[0] + "x" + in.getImageDimensions()[1]);
        }
        j.put("recording", recording).put("saved", saved);
        if (log) Log.i(TAG, j.toString());
        final String text = "ARCore: 動作中 ／ 深度: " + (depthAuto ? "使えます" : depthRaw ? "生の深度のみ" : "使えません")
                + "\n写真の大きさ: " + cameraConfigNote
                + "\n" + trackLine + "\n" + depthLine + "\n" + rawLine
                + "\n記録: " + (recording ? (tooFast ? saved + " 枚 ／ 動きが速いので保存を待っています。ゆっくり動かしてください" : saved + " 枚（ゆっくり向きを変えると自動で追加）") : saved > 0 ? saved + " 枚を保存しました" : "まだ始めていません");
        runOnUiThread(() -> statusView.setText(text));
    }

    // ------------------------------------------------------------------ 表示

    private void fail(String userMessage, Exception e) {
        Log.e(TAG, "{\"event\":\"error\",\"type\":\"" + e.getClass().getSimpleName() + "\"}", e);
        showError(userMessage);
    }

    private void showError(String msg) {
        runOnUiThread(() -> {
            errorView.setText(msg);
            errorView.setVisibility(View.VISIBLE);
            errorView.setOnClickListener(v -> errorView.setVisibility(View.GONE));
        });
    }

    private void hideError() {
        runOnUiThread(() -> errorView.setVisibility(View.GONE));
    }

    private static FloatBuffer floats(float[] a) {
        FloatBuffer f = ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        f.put(a).position(0);
        return f;
    }
}
