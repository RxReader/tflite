import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// File I/O utilities.
class FileUtil {
  const FileUtil._();

  /// Loads labels from the label file into a list of strings.
  ///
  /// A legal label file is the plain text file whose contents are split into lines, and each line
  /// is an individual value.
  static Future<List<String>> loadLabelsFromAsset(String assetName, {
    AssetBundle? bundle,
    String? package,
    Encoding encoding = utf8,
  }) async {
    final String keyName = package == null ? assetName : 'packages/$package/$assetName';
    final AssetBundle chosenBundle = bundle ?? rootBundle;
    final ByteData data = await chosenBundle.load(keyName);
    Future<String> readAsString() async {
      final Uint8List bytes = data.buffer.asUint8List();
      return utf8.decode(bytes);
    }
    if (data.lengthInBytes < 64 * 1024) {
      return _decodeLabels(readAsString);
    }
    return compute(_decodeLabels, readAsString);
  }

  /// Loads labels from the label file into a list of strings.
  ///
  /// A legal label file is the plain text file whose contents are split into lines, and each line
  /// is an individual value.
  static Future<List<String>> loadLabelsFromFile(File file, {
    Encoding encoding = utf8,
  }) async {
    Future<String> readAsString() async {
      return file.readAsString(encoding: encoding);
    }
    if (file.lengthSync() < 64 * 1024) {
      return _decodeLabels(readAsString);
    }
    return compute(_decodeLabels, readAsString);
  }

  static Future<List<String>> _decodeLabels(AsyncValueGetter<String> readAsString) async {
    final String labels = await readAsString();
    return labels.split('\n').map((String element) => element.trim()).where((String element) => element.isNotEmpty).toList();
  }
}
