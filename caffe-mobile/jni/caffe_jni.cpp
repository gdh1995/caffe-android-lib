#include <jni.h>
#include <android/log.h>
#include <string>
#include <unistd.h>

#include "caffe_mobile.hpp"

#define  LOG_TAG    "CaffeMobile"
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG, __VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)

extern "C" {

#define jni_funcname(name) Java_com_caffe_android_CaffeMobile_##name

using caffe::string;
using caffe::vector;

caffe::CaffeMobile *caffe_mobile = nullptr;

int getTimeSec();

static int pfd[2];
static pthread_t thr;
static const char *tag = "stderr";

static void *thread_func(void*) {
    ssize_t rdsz;
    char buf[1024];
    while ((rdsz = read(pfd[0], buf, sizeof(buf) - 1)) > 0) {
        buf[rdsz] = 0;  // add null-terminator
        __android_log_write(ANDROID_LOG_DEBUG, tag, buf);
    }
    return 0;
}

static int start_logger() {
    /* make stdout line-buffered and stderr unbuffered */
    // setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    pipe(pfd);
    // dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    if (pthread_create(&thr, 0, thread_func, 0) == -1)
        return -1;
    pthread_detach(thr);
    return 0;
}


void JNIEXPORT JNICALL
jni_funcname(enableLog)(JNIEnv* env, jobject thiz, jboolean enabled)
{
    start_logger();
    caffe::LogMessage::Enable(enabled != JNI_FALSE);
}

jint JNIEXPORT JNICALL
jni_funcname(loadModelOnce)(JNIEnv* env, jobject thiz, jstring modelPath, jstring weightsPath)
{
    delete caffe_mobile;
    const char *model_path = env->GetStringUTFChars(modelPath, 0);
    const char *weights_path = env->GetStringUTFChars(weightsPath, 0);
    const string model_path_str(model_path);
    const int slash_pos = model_path_str.find_last_of('/');
    if (slash_pos > 0) {
      const string new_dir = model_path_str.substr(0, slash_pos);
      chdir(new_dir.c_str());
      LOGW("%s: %s", "change current directory to", new_dir.c_str());
    }
    caffe_mobile = new caffe::CaffeMobile(model_path_str, weights_path);
    env->ReleaseStringUTFChars(modelPath, model_path);
    env->ReleaseStringUTFChars(weightsPath, weights_path);
    return 0;
}

jfloatArray JNIEXPORT JNICALL
jni_funcname(predict)(JNIEnv* env, jobject thiz)
{
    CHECK(caffe_mobile != NULL);
    vector<float> top = caffe_mobile->predict();
    LOGV("Caffe: top's count is: %d.", top.size());

    int count = top.size();
    jfloatArray ret_arr = env->NewFloatArray(count);
    env->SetFloatArrayRegion(ret_arr, 0, count, top.data());
    return ret_arr;
}

jintArray JNIEXPORT JNICALL
jni_funcname(predictTopK)(JNIEnv* env, jobject thiz, jstring imgPath, jint K)
{
    CHECK(caffe_mobile != NULL);
    const char *img_path = env->GetStringUTFChars(imgPath, 0);
    vector<int> top_k = caffe_mobile->predict_top_k(img_path, K);
    LOGD("top-1 result: %d", top_k[0]);

    K = top_k.size();
    jintArray ret_arr = env->NewIntArray(K);
    env->SetIntArrayRegion(ret_arr, 0, K, top_k.data());
    env->ReleaseStringUTFChars(imgPath, img_path);
    return ret_arr;
}

jint JNIEXPORT JNICALL
jni_funcname(setImages)(JNIEnv* env, jobject thiz, jobjectArray imgPaths)
{
    CHECK(caffe_mobile != NULL);

    int count = env->GetArrayLength(imgPaths);
    vector<string> img_paths;
    for (int i = 0; i < count; i++) {
        jstring imgPath =(jstring) (env->GetObjectArrayElement(imgPaths, i));
        const char *img_path = env->GetStringUTFChars(imgPath, 0);
        img_paths.push_back(img_path);
        env->ReleaseStringUTFChars(imgPath, img_path);
    }
    LOGD("image count: %d", img_paths.size());

    return caffe_mobile->set_images(img_paths);
}

jint JNIEXPORT JNICALL
jni_funcname(getOutputHeight)(JNIEnv* env, jobject thiz)
{
    CHECK(caffe_mobile != NULL);
    return caffe_mobile->output_height();
}

jint JNIEXPORT JNICALL
jni_funcname(getOutputNum)(JNIEnv* env, jobject thiz)
{
    CHECK(caffe_mobile != NULL);
    return caffe_mobile->output_num();
}

jdouble JNIEXPORT JNICALL
jni_funcname(getTestTime)(JNIEnv* env, jobject thiz)
{
    CHECK(caffe_mobile != NULL);
    return caffe_mobile->test_time();
}

int getTimeSec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int) now.tv_sec;
}
/*
JavaVM *g_jvm = NULL;
jobject g_obj = NULL;

void JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_MainActivity_MainActivity_setJNIEnv(JNIEnv* env, jobject obj)
{
    env->GetJavaVM(&g_jvm);
    g_obj = env->NewGlobalRef(obj);
}
*/
jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    JNIEnv* env = NULL;
    jint result = -1;

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        LOGE("GetEnv failed!");
        return result;
    }

    return JNI_VERSION_1_6;
}

int main(int argc, char const *argv[])
{
    string usage("usage: main <model> <weights> <img>");
    if (argc < 4) {
        std::cerr << usage << std::endl;
        return 1;
    }

    caffe::LogMessage::Enable(true); // enable logging
    caffe_mobile = new caffe::CaffeMobile(string(argv[1]), string(argv[2]));
    vector<int> top_3 = caffe_mobile->predict_top_k(string(argv[3]));
    for (auto k : top_3) {
        std::cout << k << std::endl;
    }
    return 0;
}

}
