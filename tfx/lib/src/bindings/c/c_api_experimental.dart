import 'dart:ffi';

import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteLib;

/// [tensorflow#c_api_experimental](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/c/c_api_experimental.h)

/// Enable or disable the NN API delegate for the interpreter (true to enable).
///
/// WARNING: This is an experimental API and subject to change.
final void Function(Pointer<TfLiteInterpreterOptions> options, bool enable) TfLiteInterpreterOptionsSetUseNNAPI =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreterOptions> options, Bool enable)>>('TfLiteInterpreterOptionsSetUseNNAPI').asFunction();

/// Set if buffer handle output is allowed.
///
/// When using hardware delegation, Interpreter will make the data of output
/// tensors available in `tensor->data` by default. If the application can
/// consume the buffer handle directly (e.g. reading output from OpenGL
/// texture), it can set this flag to false, so Interpreter won't copy the
/// data from buffer handle to CPU memory. WARNING: This is an experimental
/// API and subject to change.
final void Function(Pointer<TfLiteInterpreter> interpreter, bool allow_buffer_handle_output) TfLiteSetAllowBufferHandleOutput =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreter> interpreter, Bool allow_buffer_handle_output)>>('TfLiteSetAllowBufferHandleOutput').asFunction();
