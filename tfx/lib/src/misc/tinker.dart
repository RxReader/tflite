import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class Tinker {
  const Tinker._();

  static const MethodChannel _channel = MethodChannel('v7lin.github.io/tflite');

  static Future<void> applyWorkaroundOnOldAndroidVersions() async {
    assert(Platform.isAndroid);
    try {
      DynamicLibrary.open('libtensorflowlite_jni.so');
      DynamicLibrary.open('libtensorflowlite_gpu_jni.so');
    } on ArgumentError catch (e) {
      if (kDebugMode) {
        print(e);
      }
      // ffi 在部分 Android 5.x/6.x 机器存在兼容问题
      // https://github.com/simolus3/moor/issues/895
      // 目测和 [android:extractNativeLibs](https://developer.android.com/guide/topics/manifest/application-element) 有关
      // 插件方式
      try {
        await _channel.invokeMethod<void>('doesnt_matter');
        DynamicLibrary.open('libtensorflowlite_jni.so');
        DynamicLibrary.open('libtensorflowlite_gpu_jni.so');
        return;
      } catch (e) {
        if (kDebugMode) {
          print(e);
        }
      }
      // 其他方式
      try {
        final Uint8List appIdAsBytes = File('/proc/self/cmdline').readAsBytesSync();
        // app id ends with the first \0 character in here.
        final int endOfAppId = math.max(appIdAsBytes.indexOf(0), 0);
        final String appId = String.fromCharCodes(appIdAsBytes.sublist(0, endOfAppId));
        DynamicLibrary.open('/data/data/$appId/lib/libtensorflowlite_jni.so');
        DynamicLibrary.open('/data/data/$appId/lib/libtensorflowlite_gpu_jni.so');
        return;
      } catch (e) {
        if (kDebugMode) {
          print(e);
        }
      }
      rethrow;
    }
  }
}
