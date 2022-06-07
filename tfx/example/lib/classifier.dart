import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tfx/tflite.dart';

class Classifier {
  Classifier();

  // Maximum length of sentence
  final int _sentenceLen = 256;

  static const String _start = '<START>';
  static const String _pad = '<PAD>';
  static const String _unk = '<UNKNOWN>';

  late Map<String, int> _dict;

  // TensorFlow Lite Interpreter object
  late Interpreter _interpreter;
  bool _inited = false;

  Future<void> init() async {
    // Load model when the classifier is initialized.
    if (_inited) {
      return;
    }
    await _loadModel();
    await _loadDictionary();
    _inited = true;
  }

  Future<void> _loadModel() async {
    final ByteData byteData = await rootBundle.load('assets/text_classification.tflite');
    final Uint8List buffer = byteData.buffer.asUint8List();
    _interpreter = Interpreter.fromBuffer(buffer);
    if (kDebugMode) {
      print('Interpreter loaded successfully');
    }
  }

  Future<void> _loadDictionary() async {
    final String vocab = await rootBundle.loadString('assets/text_classification_vocab.txt');
    final Map<String, int> dict = <String, int>{};
    final List<String> vocabList = vocab.split('\n');
    for (int i = 0; i < vocabList.length; i++) {
      final List<String> entry = vocabList[i].trim().split(' ');
      dict[entry[0]] = int.parse(entry[1]);
    }
    _dict = dict;
    if (kDebugMode) {
      print('Dictionary loaded successfully');
    }
  }

  /// Isolate
  Future<List<double>> classify(String rawText) async {
    // tokenizeInputText returns List<List<double>>
    // of shape [1, 256].
    final List<List<double>> input = _tokenizeInputText(rawText);

    // output of shape [1,2].
    final List<dynamic> output = List<double>.filled(2, 0).reshape<double>(<int>[1, 2]);

    // The run method will run inference and
    // store the resulting values in output.
    _interpreter.run(input, output);

    return <double>[(output[0] as List<double>)[0], (output[0] as List<double>)[1]];
  }

  List<List<double>> _tokenizeInputText(String text) {
    // Whitespace tokenization
    final List<String> toks = text.split(' ');

    // Create a list of length==_sentenceLen filled with the value <pad>
    final List<double> vec = List<double>.filled(_sentenceLen, _dict[_pad]!.toDouble());

    int index = 0;
    if (_dict.containsKey(_start)) {
      vec[index++] = _dict[_start]!.toDouble();
    }

    // For each word in sentence find corresponding index in dict
    for (String tok in toks) {
      if (index > _sentenceLen) {
        break;
      }
      vec[index++] = _dict.containsKey(tok) ? _dict[tok]!.toDouble() : _dict[_unk]!.toDouble();
    }

    // returning List<List<double>> as our interpreter input tensor expects the shape, [1,256]
    return <List<double>>[vec];
  }

  /// Isolate
  Future<void> dispose() async {
    _interpreter.delete();
  }
}
