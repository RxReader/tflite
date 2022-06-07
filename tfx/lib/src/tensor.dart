import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/misc/list_shape.dart';
import 'package:tfx/src/quanitzation_params.dart';

class Tensor {
  Tensor(Pointer<TfLiteTensor> ref): _ref = ref {
    checkArgument(_ref.address != nullptr.address, message: 'Unable to create Tensor.');
  }

  final Pointer<TfLiteTensor> _ref;

  int /*TfLiteType*/ get type => TfLiteTensorType(_ref);

  int get numDims => TfLiteTensorNumDims(_ref);

  List<int> get shape => List<int>.generate(TfLiteTensorNumDims(_ref), (int index) => TfLiteTensorDim(_ref, index));

  int get byteSize => TfLiteTensorByteSize(_ref);

  Uint8List get data {
    final Pointer<Uint8> data = TfLiteTensorData(_ref).cast();
    checkState(data.address != nullptr.address, message: 'Tensor data is null.');
    return UnmodifiableUint8ListView(data.asTypedList(TfLiteTensorByteSize(_ref)));
  }

  set data(Uint8List bytes) {
    final int byteSize = TfLiteTensorByteSize(_ref);
    checkArgument(byteSize == bytes.length);
    final Pointer<Uint8> data = TfLiteTensorData(_ref).cast();
    checkState(data.address != nullptr.address, message: 'Tensor data is null.');
    final Uint8List externalTypedData = data.asTypedList(byteSize);
    externalTypedData.setRange(0, byteSize, bytes);
  }

  String get name => TfLiteTensorName(_ref).toDartString();

  QuantizationParams get quantizationParams {
    final TfLiteQuantizationParams params = TfLiteTensorQuantizationParams(_ref);
    return QuantizationParams(params.scale, params.zero_point);
  }

  // TODO: 着重验证 size_t
  void copyFromBuffer(Uint8List buffer) {
    final int size = buffer.length;
    final Pointer<Uint8> ptr = calloc<Uint8>(size);
    checkState(ptr.address != nullptr.address, message: 'unallocated');
    final Uint8List externalTypedData = ptr.asTypedList(size);
    externalTypedData.setRange(0, buffer.length, buffer);
    checkState(TfLiteTensorCopyFromBuffer(_ref, ptr.cast(), buffer.length) == TfLiteStatus.kTfLiteOk);
    calloc.free(ptr);
  }

  // TODO: 着重验证 size_t
  Uint8List copyToBuffer() {
    final int size = TfLiteTensorByteSize(_ref);
    final Pointer<Uint8> ptr = calloc<Uint8>(size);
    checkState(ptr.address != nullptr.address, message: 'unallocated');
    final Uint8List externalTypedData = ptr.asTypedList(size);
    checkState(TfLiteTensorCopyToBuffer(_ref, ptr.cast(), size) == TfLiteStatus.kTfLiteOk);
    // Clone the data, because once `free(ptr)`, `externalTypedData` will be volatile
    final Uint8List buffer = externalTypedData.sublist(0);
    calloc.free(ptr);
    return buffer;
  }
}

extension TensorShape on Tensor {
  List<int>? getInputShapeIfDifferent(Object? input) {
    if (input == null) {
      return null;
    }
    if (input is ByteBuffer || input is Uint8List) {
      return null;
    }
    final List<int> inputShape = _computeShapeOf(input);
    if (inputShape == shape) {
      return null;
    }
    return inputShape;
  }

  static List<int> _computeShapeOf(Object o) {
    final int size = _computeNumDims(o);
    final List<int> dims = List<int>.filled(size, 0);
    _fillShape(o, 0, dims);
    return dims;
  }

  static int _computeNumDims(Object? o) {
    if (o == null || o is! List) {
      return 0;
    }
    if (o.isEmpty) {
      throw ArgumentError('Array lengths cannot be 0.');
    }
    return 1 + _computeNumDims(o[0]);
  }

  static void _fillShape(Object o, int dim, List<int>? shape) {
    if (shape == null || dim == shape.length) {
      return;
    }
    final int len = (o as List<Object>).length;
    if (shape[dim] == 0) {
      shape[dim] = len;
    } else if (shape[dim] != len) {
      throw ArgumentError('Mismatched lengths ${shape[dim]} and $len in dimension $dim');
    }
    for (int i = 0; i < len; i++/*++i*/) {
      _fillShape(o[0], dim + 1, shape);
    }
  }
}

extension TensorCopyFrom on Tensor {
  void copyFrom(Object src) {
    copyFromBuffer(_convertObjectToBuffer(src, type));
  }

  static Uint8List _convertObjectToBuffer(Object o, int/*TfLiteType*/ tfliteType) {
    if (o is Uint8List) {
      return o;
    }
    if (o is ByteBuffer) {
      return o.asUint8List();
    }
    if (o is List<Object>) {
      final List<int> bytes = <int>[];
      for (Object e in o) {
        bytes.addAll(_convertObjectToBuffer(e, tfliteType));
      }
      return Uint8List.fromList(bytes);
    } else {
      return _convertElementToBuffer(o, tfliteType);
    }
  }

  static Uint8List _convertElementToBuffer(Object o, int/*TfLiteType*/ type) {
    if (type == TfLiteType.kTfLiteFloat32) {
      if (o is double) {
        final ByteBuffer buffer = Uint8List(4).buffer;
        ByteData.view(buffer).setFloat32(0, o, Endian.little);
        return buffer.asUint8List();
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is float32');
      }
    } else if (type == TfLiteType.kTfLiteInt32) {
      if (o is int) {
        final ByteBuffer buffer = Uint8List(4).buffer;
        ByteData.view(buffer).setInt32(0, o, Endian.little);
        return buffer.asUint8List();
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is int32');
      }
    } else if (type == TfLiteType.kTfLiteInt64) {
      if (o is int) {
        final ByteBuffer buffer = Uint8List(8).buffer;
        ByteData.view(buffer).setInt64(0, o /*, Endian.big*/);
        return buffer.asUint8List();
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is int64');
      }
    } else if (type == TfLiteType.kTfLiteInt16) {
      if (o is int) {
        final ByteBuffer buffer = Uint8List(2).buffer;
        ByteData.view(buffer).setInt16(0, o, Endian.little);
        return buffer.asUint8List();
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is int16');
      }
    } else if (type == TfLiteType.kTfLiteInt8) {
      if (o is int) {
        final ByteBuffer buffer = Uint8List(1).buffer;
        ByteData.view(buffer).setInt8(0, o);
        return buffer.asUint8List();
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is int8');
      }
    } else if (type == TfLiteType.kTfLiteFloat16) {
      if (o is double) {
        final ByteBuffer buffer = Uint8List(4).buffer;
        ByteData.view(buffer).setFloat32(0, o, Endian.little);
        return buffer.asUint8List().sublist(0, 2);
      } else {
        throw ArgumentError('The input element is ${o.runtimeType} while tensor data tfliteType is float16');
      }
    } else {
      throw ArgumentError('The input data tfliteType ${o.runtimeType} is unsupported');
    }
  }
}

extension TensorCopyTo on Tensor {
  void copyTo(Object dst) {
    final Uint8List bytes = copyToBuffer();
    if (dst is ByteBuffer) {
      final ByteData byteData = dst.asByteData();
      for (int i = 0; i < byteData.lengthInBytes; i++) {
        byteData.setUint8(i, bytes[i]);
      }
    } else {
      late Object obj;
      if (dst is Uint8List) {
        obj = bytes;
      } else {
        obj = _convertBufferToObject(bytes, type, shape);
      }
      if (obj is List && dst is List) {
        _duplicateList(obj, dst);
      } else {
        throw UnsupportedError('${dst.runtimeType} is not Supported.');
      }
    }
  }

  Object _convertBufferToObject(Uint8List bytes, int/*TfLiteType*/ type, List<int> shape) {
    final List<dynamic> list = <dynamic>[];
    if (type == TfLiteType.kTfLiteFloat32) {
      for (int i = 0; i < bytes.length; i += 4) {
        list.add(ByteData.view(bytes.buffer).getFloat32(i, Endian.little));
      }
      return list.reshape<double>(shape);
    } else if (type == TfLiteType.kTfLiteInt32) {
      for (int i = 0; i < bytes.length; i += 4) {
        list.add(ByteData.view(bytes.buffer).getInt32(i, Endian.little));
      }
      return list.reshape<int>(shape);
    } else if (type == TfLiteType.kTfLiteInt64) {
      for (int i = 0; i < bytes.length; i += 8) {
        list.add(ByteData.view(bytes.buffer).getInt64(i));
      }
      return list.reshape<int>(shape);
    } else if (type == TfLiteType.kTfLiteInt16) {
      for (int i = 0; i < bytes.length; i += 2) {
        list.add(ByteData.view(bytes.buffer).getInt16(i, Endian.little));
      }
      return list.reshape<int>(shape);
    } else if (type == TfLiteType.kTfLiteFloat16) {
      final Uint8List list32 = Uint8List(bytes.length * 2);
      for (int i = 0; i < bytes.length; i += 2) {
        list32[i] = bytes[i];
        list32[i + 1] = bytes[i + 1];
      }
      for (int i = 0; i < list32.length; i += 4) {
        list.add(ByteData.view(list32.buffer).getFloat32(i, Endian.little));
      }
      return list.reshape<double>(shape);
    } else if (type == TfLiteType.kTfLiteInt8) {
      for (int i = 0; i < bytes.length; i += 1) {
        list.add(ByteData.view(bytes.buffer).getInt8(i));
      }
      return list.reshape<int>(shape);
    }
    throw UnsupportedError('$type is not Supported.');
  }

  void _duplicateList(List<dynamic> obj, List<dynamic> dst) {
    final List<int> objShape = obj.shape;
    final List<int> dstShape = dst.shape;
    bool equal = true;
    if (objShape.length == dst.shape.length) {
      for (int i = 0; i < objShape.length; i++) {
        if (objShape[i] != dstShape[i]) {
          equal = false;
          break;
        }
      }
    } else {
      equal = false;
    }
    if (!equal) {
      throw ArgumentError('Output object shape mismatch, interpreter returned output of shape: ${obj.shape} while shape of output provided as argument in run is: ${dst.shape}');
    }
    for (int i = 0; i < obj.length; i++) {
      dst[i] = obj[i];
    }
  }
}
