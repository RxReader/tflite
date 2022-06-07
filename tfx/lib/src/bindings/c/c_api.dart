import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteLib;

/// [tensorflow#c_api](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/c/c_api.h)

/// --------------------------------------------------------------------------
/// Opaque types used by the C API.

/// TfLiteModel wraps a loaded TensorFlow Lite model.
class TfLiteModel extends Opaque {}

/// TfLiteInterpreterOptions allows customized interpreter configuration.
class TfLiteInterpreterOptions extends Opaque {}

/// Allows delegation of nodes to alternative backends.
class TfLiteDelegate extends Opaque {}

/// TfLiteInterpreter provides inference from a provided model.
class TfLiteInterpreter extends Opaque {}

/// A tensor in the interpreter system which is a wrapper around a buffer of
/// data including a dimensionality (or NULL if not currently defined).
class TfLiteTensor extends Opaque {}

/// --------------------------------------------------------------------------
/// TfLiteVersion returns a string describing version information of the
/// TensorFlow Lite library. TensorFlow Lite uses semantic versioning.
final Pointer<Utf8> Function() TfLiteVersion = tfliteLib.lookup<NativeFunction<Pointer<Utf8> Function()>>('TfLiteVersion').asFunction();

/// Returns a model from the provided buffer, or null on failure.
///
/// NOTE: The caller retains ownership of the `model_data` and should ensure that
/// the lifetime of the `model_data` must be at least as long as the lifetime
/// of the `TfLiteModel`.
final Pointer<TfLiteModel> Function(Pointer<Void> model_data, int model_size) TfLiteModelCreate =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteModel> Function(Pointer<Void> model_data, Size model_size)>>('TfLiteModelCreate').asFunction();

/// Returns a model from the provided file, or null on failure.
final Pointer<TfLiteModel> Function(Pointer<Utf8> model_path) TfLiteModelCreateFromFile =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteModel> Function(Pointer<Utf8> model_path)>>('TfLiteModelCreateFromFile').asFunction();

/// Destroys the model instance.
final void Function(Pointer<TfLiteModel> model) TfLiteModelDelete = tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteModel> model)>>('TfLiteModelDelete').asFunction();

/// Returns a new interpreter options instances.
final Pointer<TfLiteInterpreterOptions> Function() TfLiteInterpreterOptionsCreate =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteInterpreterOptions> Function()>>('TfLiteInterpreterOptionsCreate').asFunction();

/// Destroys the interpreter options instance.
final void Function(Pointer<TfLiteInterpreterOptions> options) TfLiteInterpreterOptionsDelete =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreterOptions> options)>>('TfLiteInterpreterOptionsDelete').asFunction();

/// Sets the number of CPU threads to use for the interpreter.
final void Function(Pointer<TfLiteInterpreterOptions> options, int num_threads) TfLiteInterpreterOptionsSetNumThreads =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreterOptions> options, Int32 num_threads)>>('TfLiteInterpreterOptionsSetNumThreads').asFunction();

/// Adds a delegate to be applied during `TfLiteInterpreter` creation.
///
/// If delegate application fails, interpreter creation will also fail with an
/// associated error logged.
///
/// NOTE: The caller retains ownership of the delegate and should ensure that it
/// remains valid for the duration of any created interpreter's lifetime.
final void Function(Pointer<TfLiteInterpreterOptions> options, Pointer<TfLiteDelegate> delegate) TfLiteInterpreterOptionsAddDelegate =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreterOptions> options, Pointer<TfLiteDelegate> delegate)>>('TfLiteInterpreterOptionsAddDelegate').asFunction();

/// Sets a custom error reporter for interpreter execution.
///
/// * `reporter` takes the provided `user_data` object, as well as a C-style
///   format string and arg list (see also vprintf).
/// * `user_data` is optional. If non-null, it is owned by the client and must
///   remain valid for the duration of the interpreter lifetime.
final void Function(Pointer<TfLiteInterpreterOptions> options, Pointer<NativeFunction<Void Function(Pointer<Void> user_data, Pointer<Utf8> format, /*va_list*/ Pointer<Void> args)>> reporter,
        Pointer<Void> user_data) TfLiteInterpreterOptionsSetErrorReporter =
    tfliteLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<TfLiteInterpreterOptions> options, Pointer<NativeFunction<Void Function(Pointer<Void> user_data, Pointer<Utf8> format, /*va_list*/ Pointer<Void> args)>> reporter,
                    Pointer<Void> user_data)>>('TfLiteInterpreterOptionsSetErrorReporter')
        .asFunction();

/// Returns a new interpreter using the provided model and options, or null on
/// failure.
///
/// * `model` must be a valid model instance. The caller retains ownership of the
///   object, and the model must outlive the interpreter.
/// * `optional_options` may be null. The caller retains ownership of the object,
///   and can safely destroy it immediately after creating the interpreter.
///
/// NOTE: The client *must* explicitly allocate tensors before attempting to
/// access input tensor data or invoke the interpreter.
final Pointer<TfLiteInterpreter> Function(Pointer<TfLiteModel> model, Pointer<TfLiteInterpreterOptions> optional_options) TfLiteInterpreterCreate =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteInterpreter> Function(Pointer<TfLiteModel> model, Pointer<TfLiteInterpreterOptions> optional_options)>>('TfLiteInterpreterCreate').asFunction();

/// Destroys the interpreter.
final void Function(Pointer<TfLiteInterpreter> interpreter) TfLiteInterpreterDelete =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteInterpreter> interpreter)>>('TfLiteInterpreterDelete').asFunction();

/// Returns the number of input tensors associated with the model.
final int Function(Pointer<TfLiteInterpreter> interpreter) TfLiteInterpreterGetInputTensorCount =
    tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteInterpreter> interpreter)>>('TfLiteInterpreterGetInputTensorCount').asFunction();

/// Returns the tensor associated with the input index.
/// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
final Pointer<TfLiteTensor> Function(Pointer<TfLiteInterpreter> interpreter, int input_index) TfLiteInterpreterGetInputTensor =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteTensor> Function(Pointer<TfLiteInterpreter> interpreter, Int32 input_index)>>('TfLiteInterpreterGetInputTensor').asFunction();

/// Resizes the specified input tensor.
///
/// NOTE: After a resize, the client *must* explicitly allocate tensors before
/// attempting to access the resized tensor data or invoke the interpreter.
///
/// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
///
/// This function makes a copy of the input dimensions, so the client can safely
/// deallocate `input_dims` immediately after this function returns.
final int /*TfLiteStatus*/ Function(Pointer<TfLiteInterpreter> interpreter, int input_index, Pointer<Int32> input_dims, int input_dims_size) TfLiteInterpreterResizeInputTensor = tfliteLib
    .lookup<NativeFunction<Int32 /*TfLiteStatus*/ Function(Pointer<TfLiteInterpreter> interpreter, Int32 input_index, Pointer<Int32> input_dims, Int32 input_dims_size)>>(
        'TfLiteInterpreterResizeInputTensor')
    .asFunction();

/// Updates allocations for all tensors, resizing dependent tensors using the
/// specified input tensor dimensionality.
///
/// This is a relatively expensive operation, and need only be called after
/// creating the graph and/or resizing any inputs.
final int /*TfLiteStatus*/ Function(Pointer<TfLiteInterpreter> interpreter) TfLiteInterpreterAllocateTensors = tfliteLib
    .lookup<
        NativeFunction<
            Int32 /*TfLiteStatus*/
                Function(Pointer<TfLiteInterpreter> interpreter)>>('TfLiteInterpreterAllocateTensors')
    .asFunction();

/// Runs inference for the loaded graph.
///
/// Before calling this function, the caller should first invoke
/// TfLiteInterpreterAllocateTensors() and should also set the values for the
/// input tensors.  After successfully calling this function, the values for the
/// output tensors will be set.
///
/// NOTE: It is possible that the interpreter is not in a ready state to
/// evaluate (e.g., if AllocateTensors() hasn't been called, or if a
/// ResizeInputTensor() has been performed without a subsequent call to
/// AllocateTensors()).
///
///   If the (experimental!) delegate fallback option was enabled in the
///   interpreter options, then the interpreter will automatically fall back to
///   not using any delegates if execution with delegates fails. For details, see
///   TfLiteInterpreterOptionsSetEnableDelegateFallback in c_api_experimental.h.
///
/// Returns one of the following status codes:
///  - kTfLiteOk: Success. Output is valid.
///  - kTfLiteDelegateError: Execution with delegates failed, due to a problem
///    with the delegate(s). If fallback was not enabled, output is invalid.
///    If fallback was enabled, this return value indicates that fallback
///    succeeded, the output is valid, and all delegates previously applied to
///    the interpreter have been undone.
///  - kTfLiteApplicationError: Same as for kTfLiteDelegateError, except that
///    the problem was not with the delegate itself, but rather was
///    due to an incompatibility between the delegate(s) and the
///    interpreter or model.
///  - kTfLiteError: Unexpected/runtime failure. Output is invalid.
final int /*TfLiteStatus*/ Function(Pointer<TfLiteInterpreter> interpreter) TfLiteInterpreterInvoke =
    tfliteLib.lookup<NativeFunction<Int32 /*TfLiteStatus*/ Function(Pointer<TfLiteInterpreter> interpreter)>>('TfLiteInterpreterInvoke').asFunction();

/// Returns the number of output tensors associated with the model.
final int Function(Pointer<TfLiteInterpreter> interpreter) TfLiteInterpreterGetOutputTensorCount =
    tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteInterpreter> interpreter)>>('TfLiteInterpreterGetOutputTensorCount').asFunction();

/// Returns the tensor associated with the output index.
/// REQUIRES: 0 <= output_index < TfLiteInterpreterGetOutputTensorCount(tensor)
///
/// NOTE: The shape and underlying data buffer for output tensors may be not
/// be available until after the output tensor has been both sized and allocated.
/// In general, best practice is to interact with the output tensor *after*
/// calling TfLiteInterpreterInvoke().
final Pointer<TfLiteTensor> Function(Pointer<TfLiteInterpreter> interpreter, int output_index) TfLiteInterpreterGetOutputTensor =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteTensor> Function(Pointer<TfLiteInterpreter> interpreter, Int32 output_index)>>('TfLiteInterpreterGetOutputTensor').asFunction();

/// --------------------------------------------------------------------------
/// TfLiteTensor wraps data associated with a graph tensor.
///
/// Note that, while the TfLiteTensor struct is not currently opaque, and its
/// fields can be accessed directly, these methods are still convenient for
/// language bindings. In the future the tensor struct will likely be made opaque
/// in the public API.

/// Returns the type of a tensor element.
final int /*TfLiteType*/ Function(Pointer<TfLiteTensor> tensor) TfLiteTensorType =
    tfliteLib.lookup<NativeFunction<Int32 /*TfLiteType*/ Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorType').asFunction();

/// Returns the number of dimensions that the tensor has.
final int Function(Pointer<TfLiteTensor> tensor) TfLiteTensorNumDims = tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorNumDims').asFunction();

/// Returns the length of the tensor in the "dim_index" dimension.
/// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
final int Function(Pointer<TfLiteTensor> tensor, int dim_index) TfLiteTensorDim =
    tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteTensor> tensor, Int32 dim_index)>>('TfLiteTensorDim').asFunction();

/// Returns the size of the underlying data in bytes.
final int Function(Pointer<TfLiteTensor> tensor) TfLiteTensorByteSize = tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorByteSize').asFunction();

/// Returns a pointer to the underlying data buffer.
///
/// NOTE: The result may be null if tensors have not yet been allocated, e.g.,
/// if the Tensor has just been created or resized and `TfLiteAllocateTensors()`
/// has yet to be called, or if the output tensor is dynamically sized and the
/// interpreter hasn't been invoked.
final Pointer<Void> Function(Pointer<TfLiteTensor> tensor) TfLiteTensorData = tfliteLib.lookup<NativeFunction<Pointer<Void> Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorData').asFunction();

/// Returns the (null-terminated) name of the tensor.
final Pointer<Utf8> Function(Pointer<TfLiteTensor> tensor) TfLiteTensorName = tfliteLib.lookup<NativeFunction<Pointer<Utf8> Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorName').asFunction();

/// Returns the parameters for asymmetric quantization. The quantization
/// parameters are only valid when the tensor type is `kTfLiteUInt8` and the
/// `scale != 0`. Quantized values can be converted back to float using:
///    real_value = scale * (quantized_value - zero_point);
final TfLiteQuantizationParams Function(Pointer<TfLiteTensor> tensor) TfLiteTensorQuantizationParams =
    tfliteLib.lookup<NativeFunction<TfLiteQuantizationParams Function(Pointer<TfLiteTensor> tensor)>>('TfLiteTensorQuantizationParams').asFunction();

/// Copies from the provided input buffer into the tensor's buffer.
/// REQUIRES: input_data_size == TfLiteTensorByteSize(tensor)
final int /*TfLiteStatus*/ Function(Pointer<TfLiteTensor> tensor, Pointer<Void> input_data, int input_data_size) TfLiteTensorCopyFromBuffer =
    tfliteLib.lookup<NativeFunction<Int32 /*TfLiteStatus*/ Function(Pointer<TfLiteTensor> tensor, Pointer<Void> input_data, Size input_data_size)>>('TfLiteTensorCopyFromBuffer').asFunction();

/// Copies to the provided output buffer from the tensor's buffer.
/// REQUIRES: output_data_size == TfLiteTensorByteSize(tensor)
final int /*TfLiteStatus*/ Function(Pointer<TfLiteTensor> output_tensor, Pointer<Void> output_data, int output_data_size) TfLiteTensorCopyToBuffer =
    tfliteLib.lookup<NativeFunction<Int32 Function(Pointer<TfLiteTensor> output_tensor, Pointer<Void> output_data, Size output_data_size)>>('TfLiteTensorCopyToBuffer').asFunction();

