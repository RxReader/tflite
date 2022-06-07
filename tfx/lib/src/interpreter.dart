import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/c/c_api_experimental.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/delegate.dart';
import 'package:tfx/src/model.dart';
import 'package:tfx/src/tensor.dart';

class Interpreter {
  factory Interpreter.fromFile(String modelPath, {InterpreterOptions? options}) {
    final Model model = Model.fromFile(modelPath);
    final Interpreter interpreter = Interpreter._create(model, options: options);
    model.delete();
    return interpreter;
  }

  factory Interpreter.fromBuffer(Uint8List buffer, {InterpreterOptions? options}) {
    final Model model = Model.fromBuffer(buffer);
    final Interpreter interpreter = Interpreter._create(model, options: options);
    model.delete();
    return interpreter;
  }

  factory Interpreter._create(Model model, {InterpreterOptions? options}) {
    final Pointer<TfLiteInterpreter> interpreter = TfLiteInterpreterCreate(model.ref, options?.ref ?? nullptr.cast());
    checkArgument(interpreter.address != nullptr.address, message: 'Unable to create Interpreter.');
    return Interpreter._(interpreter);
  }

  Interpreter._(Pointer<TfLiteInterpreter> ref) : _ref = ref {
    _allocateTensors();
  }

  final Pointer<TfLiteInterpreter> _ref;
  bool _allocated = false;
  bool _deleted = false;

  // FEATURE: allowFp16PrecisionForFp32

  // ignore: avoid_setters_without_getters
  set allowBufferHandleOutput(bool allowBufferHandleOutput) => TfLiteSetAllowBufferHandleOutput(_ref, allowBufferHandleOutput);

  // FEATURE: setCancelled

  int get inputTensorCount => TfLiteInterpreterGetInputTensorCount(_ref);

  Tensor getInputTensor(int index) {
    final int tensorsCount = TfLiteInterpreterGetInputTensorCount(_ref);
    if (index < 0 || index >= tensorsCount) {
      throw ArgumentError('Invalid input Tensor index: $index');
    }
    return Tensor(TfLiteInterpreterGetInputTensor(_ref, index));
  }

  List<Tensor> get inputTensors {
    return List<Tensor>.generate(
      TfLiteInterpreterGetInputTensorCount(_ref),
      (int index) => Tensor(TfLiteInterpreterGetInputTensor(_ref, index)),
      growable: false,
    );
  }

  void resizeInputTensor(int index, List<int> shape) {
    final int dimsSize = shape.length;
    final Pointer<Int32> dims = calloc<Int32>(dimsSize);
    final Int32List externalTypedData = dims.asTypedList(dimsSize);
    externalTypedData.setRange(0, dimsSize, shape);
    try {
      checkState(TfLiteInterpreterResizeInputTensor(_ref, index, dims, dimsSize) == TfLiteStatus.kTfLiteOk);
    } finally {
      calloc.free(dims);
    }
    _allocated = false;
  }

  void _allocateTensors() {
    checkState(!_allocated, message: 'Interpreter already allocated.');
    checkState(TfLiteInterpreterAllocateTensors(_ref) == TfLiteStatus.kTfLiteOk);
    _allocated = true;
  }

  void invoke() {
    checkState(_allocated, message: 'Interpreter not allocated.');
    checkState(TfLiteInterpreterInvoke(_ref) == TfLiteStatus.kTfLiteOk);
  }

  int get outputTensorCount => TfLiteInterpreterGetOutputTensorCount(_ref);

  Tensor getOutputTensor(int index) {
    final int tensorsCount = TfLiteInterpreterGetOutputTensorCount(_ref);
    if (index < 0 || index >= tensorsCount) {
      throw ArgumentError('Invalid output Tensor index: $index');
    }
    return Tensor(TfLiteInterpreterGetOutputTensor(_ref, index));
  }

  List<Tensor> get outputTensors {
    return List<Tensor>.generate(
      TfLiteInterpreterGetOutputTensorCount(_ref),
      (int index) => Tensor(TfLiteInterpreterGetOutputTensor(_ref, index)),
      growable: false,
    );
  }

  void delete() {
    checkState(!_deleted, message: 'Interpreter already deleted.');
    TfLiteInterpreterDelete(_ref);
    _deleted = true;
  }
}

extension InterpreterRunner on Interpreter {
  void run(Object input, Object output) {
    runForMultipleInputs(<Object>[input], <int, Object>{0: output});
  }

  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) {
    checkArgument(inputs.isNotEmpty, message: 'Input error: Inputs should not be null or empty.');
    checkArgument(outputs.isNotEmpty, message: 'Input error: Outputs should not be null or empty.');

    List<Tensor> inputTensors = this.inputTensors;

    for (int i = 0; i < inputs.length; i++) {
      final Tensor tensor = inputTensors[i];
      final List<int>? newShape = tensor.getInputShapeIfDifferent(inputs[i]);
      if (newShape != null) {
        resizeInputTensor(i, newShape);
      }
    }

    if (!_allocated) {
      _allocateTensors();
    }

    inputTensors = this.inputTensors;
    for (int i = 0; i < inputs.length; i++) {
      inputTensors[i].copyFrom(inputs[i]);
    }

    // final int inferenceStartNanos = DateTime.now().microsecondsSinceEpoch;
    invoke();
    // final int lastNativeInferenceDurationMicroSeconds = DateTime.now().microsecondsSinceEpoch - inferenceStartNanos;

    final List<Tensor> outputTensors = this.outputTensors;
    for (int i = 0; i < outputTensors.length; i++) {
      outputTensors[i].copyTo(outputs[i]!);
    }
  }
}

class InterpreterOptions {
  factory InterpreterOptions() {
    final Pointer<TfLiteInterpreterOptions> options = TfLiteInterpreterOptionsCreate();
    checkArgument(options.address != nullptr.address, message: 'Unable to create InterpreterOptions.');
    return InterpreterOptions._(options);
  }

  InterpreterOptions._(this._ref);

  final Pointer<TfLiteInterpreterOptions> _ref;
  bool _deleted = false;

  // ignore: avoid_setters_without_getters
  set numThreads(int numThreads) {
    TfLiteInterpreterOptionsSetNumThreads(_ref, numThreads);
  }

  /// Android
  // ignore: avoid_setters_without_getters
  set useNnApi(bool useNnApi) {
    assert(Platform.isAndroid);
    TfLiteInterpreterOptionsSetUseNNAPI(_ref, useNnApi);
  }

  void addDelegate(Delegate delegate) {
    TfLiteInterpreterOptionsAddDelegate(_ref, delegate.ref);
  }

  void delete() {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    TfLiteInterpreterOptionsDelete(_ref);
    _deleted = true;
  }
}

extension InterpreterOptionsExtension on InterpreterOptions {
  Pointer<TfLiteInterpreterOptions> get ref => _ref;
}
