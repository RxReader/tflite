import 'dart:io';

import 'package:tfx/src/misc/tinker.dart';

class TFLite {
  const TFLite._();

  static void init() {
    if (Platform.isAndroid) {
      Tinker.applyWorkaroundOnOldAndroidVersions();
    }
  }
}
