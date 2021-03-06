import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/misc/tinker.dart';

class TFLite {
  const TFLite._();

  static void init() {
    if (Platform.isAndroid) {
      Tinker.applyWorkaroundOnOldAndroidVersions();
    }
  }

  static String runtimeVersion() {
    return TfLiteVersion().toDartString();
  }
}
