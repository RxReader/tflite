import 'dart:ffi';

import 'dart:io';

final DynamicLibrary tfliteLib = () {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libtensorflowlite_jni.so');
  } else if (Platform.isIOS) {
    return DynamicLibrary.process();
  }
  throw UnsupportedError('platform(${Platform.operatingSystem}) not supported');
}();

final DynamicLibrary tfliteGpuLib = () {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libtensorflowlite_gpu_jni.so');
  } else if (Platform.isIOS) {
    return DynamicLibrary.process();
  }
  throw UnsupportedError('platform(${Platform.operatingSystem}) not supported');
}();
