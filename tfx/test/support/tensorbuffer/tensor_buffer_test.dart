import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/misc/float_loss_precision.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// Test helper class for inserting and retrieving arrays.
class ArrayTestRunner {
  ArrayTestRunner._(this.srcArrays, this.arrDataTypes, this.arrShapes, this.tensorBufferShape, this.expectedResForFloatBuf, this.expectedResForByteBuf);

  // List of TensorBuffer types to be tested.
  static const List<int> BUFFER_TYPE_LIST = <int>[TfLiteType.kTfLiteFloat32, TfLiteType.kTfLiteUInt8];

  // List of source arrays to be loaded into TensorBuffer during the tests.
  final List<Object> srcArrays;

  // List of array data type with respect to srcArrays.
  final List<int> arrDataTypes;

  // List of array shape with respect to srcArrays.
  final List<List<int>> arrShapes;
  final List<int>? tensorBufferShape;
  final ExpectedResults expectedResForFloatBuf;
  final ExpectedResults expectedResForByteBuf;

  void run() {
    for (int bufferDataType in BUFFER_TYPE_LIST) {
      late final TensorBuffer tensorBuffer;
      if (tensorBufferShape == null) {
        tensorBuffer = TensorBuffer.createDynamic(bufferDataType);
      } else {
        tensorBuffer = TensorBuffer.createFixedSize(tensorBufferShape!, bufferDataType);
      }
      for (int i = 0; i < srcArrays.length; i++) {
        switch (arrDataTypes[i]) {
          case TfLiteType.kTfLiteUInt8:
            final List<int> arrInt = srcArrays[i] as List<int>;
            tensorBuffer.loadIntArray(arrInt, arrShapes[i]);
            break;
          case TfLiteType.kTfLiteFloat32:
            final List<double> arrFloat = srcArrays[i] as List<double>;
            tensorBuffer.loadFloatArray(arrFloat, arrShapes[i]);
            break;
          default:
            break;
        }
      }
      checkResults(tensorBuffer);
    }
  }

  void checkResults(TensorBuffer tensorBuffer) {
    ExpectedResults er;
    switch (tensorBuffer.dataType) {
      case TfLiteType.kTfLiteUInt8:
        er = expectedResForByteBuf;
        break;
      case TfLiteType.kTfLiteFloat32:
        er = expectedResForFloatBuf;
        break;
      default:
        throw AssertionError('Cannot test TensorBuffer in the DataType of ${tensorBuffer.dataType}');
    }

    // Checks getIntArray() and getFloatArray().
    final List<int> resIntArr = tensorBuffer.getIntArray();
    expect(listEquals(resIntArr, er.intArr), isTrue);
    final List<double> resFloatArr = tensorBuffer.getFloatArray();
    expect(listEquals(resFloatArr, er.floatArr), isTrue);
    expect(listEquals(tensorBuffer.shape, er.shape), isTrue);

    // Checks getIntValue(int index) and getFloatValue(int index).
    final int flatSize = tensorBuffer.flatSize;
    final List<double> resFloatValues = List<double>.filled(flatSize, 0.0);
    final List<int> resIntValues = List<int>.filled(flatSize, 0);
    for (int i = 0; i < flatSize; i++) {
      resFloatValues[i] = tensorBuffer.getFloatValue(i);
      resIntValues[i] = tensorBuffer.getIntValue(i);
    }
    expect(listEquals(resFloatValues, er.floatArr), isTrue);
    expect(listEquals(resIntValues, er.intArr), isTrue);
  }
}

class ArrayTestRunnerBuilder {
  ArrayTestRunnerBuilder.newInstance();

  final List<Object> _srcArrays = <Object>[];
  final List<int> _arrDataTypes = <int>[];
  final List<List<int>> _arrShapes = <List<int>>[];
  List<int>? _tensorBufferShape;
  final ExpectedResults _expectedResForFloatBuf = ExpectedResults();
  final ExpectedResults _expectedResForByteBuf = ExpectedResults();

  /// Loads a test array into the test runner.
  void addSrcArray(List<dynamic /*int/double*/ > src, List<int> shape) {
    // src should be a primitive 1D array.
    if (src is List<double>) {
      _srcArrays.add(src);
      _arrDataTypes.add(TfLiteType.kTfLiteFloat32);
      _arrShapes.add(shape);
    } else if (src is List<int>) {
      _srcArrays.add(src);
      _arrDataTypes.add(TfLiteType.kTfLiteUInt8);
      _arrShapes.add(shape);
    } else {
      throw AssertionError('Cannot resolve srouce arrays');
    }
  }

  void setTensorBufferShape(List<int> tensorBufferShape) {
    _tensorBufferShape = tensorBufferShape;
  }

  void setExpectedResults(int /*TfLiteType*/ bufferType, List<double> expectedFloatArr, List<int> expectedIntArr) {
    late final ExpectedResults er;
    switch (bufferType) {
      case TfLiteType.kTfLiteUInt8:
        er = _expectedResForByteBuf;
        break;
      case TfLiteType.kTfLiteFloat32:
        er = _expectedResForFloatBuf;
        break;
      default:
        throw AssertionError('Cannot test TensorBuffer in the TfLiteType of $bufferType');
    }
    er.floatArr = expectedFloatArr;
    er.intArr = expectedIntArr;
  }

  ArrayTestRunner build() {
    late final List<int> expectedShape;
    if (_arrShapes.isEmpty) {
      // If no array will be loaded, the array is an empty array.
      expectedShape = <int>[0];
    } else {
      expectedShape = _arrShapes[_arrShapes.length - 1];
    }
    _expectedResForByteBuf.shape = expectedShape;
    _expectedResForFloatBuf.shape = expectedShape;
    return ArrayTestRunner._(_srcArrays, _arrDataTypes, _arrShapes, _tensorBufferShape, _expectedResForFloatBuf, _expectedResForByteBuf);
  }
}

class ExpectedResults {
  late final List<double> floatArr;
  late final List<int> intArr;
  late final List<int> shape;
}

// FLOAT_ARRAY1 and INT_ARRAY1 correspond to each other.
const List<int> ARRAY1_SHAPE = <int>[2, 3];
const List<double> FLOAT_ARRAY1 = <double>[500.1, 4.2, 3.3, 2.4, 1.5, 6.1];
const List<double> FLOAT_ARRAY1_ROUNDED = <double>[500.0, 4.0, 3.0, 2.0, 1.0, 6.0];
// FLOAT_ARRAY1_CAPPED and INT_ARRAY1_CAPPED correspond to the expected values when converted into
// uint8.
const List<double> FLOAT_ARRAY1_CAPPED = <double>[255.0, 4.0, 3.0, 2.0, 1.0, 6.0];
const List<int> INT_ARRAY1 = <int>[500, 4, 3, 2, 1, 6];
const List<int> INT_ARRAY1_CAPPED = <int>[255, 4, 3, 2, 1, 6];
// FLOAT_ARRAY2 and INT_ARRAY2 correspond to each other.
const List<int> ARRAY2_SHAPE = <int>[2, 1];
const List<double> FLOAT_ARRAY2 = <double>[6.7, 7.6];
const List<double> FLOAT_ARRAY2_ROUNDED = <double>[6.0, 7.0];
const List<int> INT_ARRAY2 = <int>[6, 7];
// FLOAT_ARRAY2 and FLOAT_ARRAY3 have the same size.
const List<int> ARRAY3_SHAPE = <int>[2, 1];
const List<double> FLOAT_ARRAY3 = <double>[8.2, 9.9];
const List<double> FLOAT_ARRAY3_ROUNDED = <double>[8.0, 9.0];
// INT_ARRAY2 and INT_ARRAY3 have the same size.
const List<int> INT_ARRAY3 = <int>[8, 9];
const List<int> EMPTY_ARRAY_SHAPE = <int>[0];
const List<int> EMPTY_INT_ARRAY = <int>[];
const List<double> EMPTY_FLOAT_ARRAY = <double>[];
// Single element array which represents a scalar.
const List<int> SCALAR_ARRAY_SHAPE = <int>[];
const List<double> FLOAT_SCALAR_ARRAY = <double>[800.2];
const List<double> FLOAT_SCALAR_ARRAY_ROUNDED = <double>[800.0];
const List<double> FLOAT_SCALAR_ARRAY_CAPPED = <double>[255.0];
const List<int> INT_SCALAR_ARRAY = <int>[800];
const List<int> INT_SCALAR_ARRAY_CAPPED = <int>[255];
// Several different ByteBuffer.
final ByteBuffer EMPTY_BYTE_BUFFER = ByteData(0).buffer;
final ByteBuffer FLOAT_BYTE_BUFFER1 = ByteData(24).buffer;

void main() {
  test('testCreateFixedSizeTensorBufferFloat', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBuffer tensorBufferFloat = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteFloat32);
    expect(tensorBufferFloat.dataType, TfLiteType.kTfLiteFloat32);
    expect(tensorBufferFloat.flatSize, 6);
  });

  test('testCreateFixedSizeTensorBufferUint8', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBuffer tensorBufferUint8 = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
    expect(tensorBufferUint8.dataType, TfLiteType.kTfLiteUInt8);
    expect(tensorBufferUint8.flatSize, 6);
  });

  test('testCreateDynamicTensorBufferFloat', () {
    final TensorBuffer tensorBufferFloat = TensorBuffer.createDynamic(TfLiteType.kTfLiteFloat32);
    expect(tensorBufferFloat.dataType, TfLiteType.kTfLiteFloat32);
  });

  test('testCreateDynamicTensorBufferUint8', () {
    final TensorBuffer tensorBufferFloat = TensorBuffer.createDynamic(TfLiteType.kTfLiteUInt8);
    expect(tensorBufferFloat.dataType, TfLiteType.kTfLiteUInt8);
  });

  test('testCreateTensorBufferFromFixedSize', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBuffer src = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteFloat32);
    expect(listEquals(dst.shape, <int>[1, 2, 3]), isTrue);
  });

  test('testCreateTensorBufferFromDynamicSize', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBuffer src = TensorBuffer.createDynamic(TfLiteType.kTfLiteUInt8);
    src.resize(shape);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteFloat32);
    expect(listEquals(dst.shape, <int>[1, 2, 3]), isTrue);
  });

  test('testCreateTensorBufferUInt8FromUInt8', () {
    final List<int> shape = <int>[INT_ARRAY1.length];
    final TensorBuffer src = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
    src.loadIntArray(INT_ARRAY1);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteUInt8);
    final List<int> data = dst.getIntArray();
    expect(listEquals(data, INT_ARRAY1_CAPPED), isTrue);
  });

  test('testCreateTensorBufferUInt8FromFloat32', () {
    final TensorBuffer src = TensorBuffer.createDynamic(TfLiteType.kTfLiteFloat32);
    src.loadFloatArray(FLOAT_ARRAY1, ARRAY1_SHAPE);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteUInt8);
    final List<int> data = dst.getIntArray();
    expect(listEquals(data, INT_ARRAY1_CAPPED), isTrue);
  });

  test('testCreateTensorBufferFloat32FromUInt8', () {
    final TensorBuffer src = TensorBuffer.createDynamic(TfLiteType.kTfLiteUInt8);
    src.loadIntArray(INT_ARRAY1, ARRAY1_SHAPE);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteFloat32);
    final List<double> data = dst.getFloatArray();
    expect(listEquals(data, FLOAT_ARRAY1_CAPPED), isTrue);
  });

  test('testCreateTensorBufferFloat32FromFloat32', () {
    final List<int> shape = <int>[FLOAT_ARRAY1.length];
    final TensorBuffer src = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteFloat32);
    src.loadFloatArray(FLOAT_ARRAY1);
    final TensorBuffer dst = TensorBuffer.createFrom(src, TfLiteType.kTfLiteFloat32);
    final List<double> data = dst.getFloatArray();
    expect(listEquals(data, FLOAT_ARRAY1.lossPrecision), isTrue);
  });

  test('testGetBuffer', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBuffer tensorBufferUint8 = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
    expect(listEquals(tensorBufferUint8.buffer.asUint8List(), List<int>.filled(tensorBufferUint8.flatSize, 0)), isTrue);
  });

  test('testLoadAndGetIntArrayWithFixedSizeForScalarArray', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(INT_SCALAR_ARRAY, SCALAR_ARRAY_SHAPE)
      ..setTensorBufferShape(SCALAR_ARRAY_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_SCALAR_ARRAY_ROUNDED,
        /*expectedIntArr=*/ INT_SCALAR_ARRAY,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_SCALAR_ARRAY_CAPPED,
        /*expectedIntArr=*/ INT_SCALAR_ARRAY_CAPPED,
      );
    builder.build().run();
  });

  test('testLoadAndGetFloatArrayWithFixedSizeForScalarArray', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(FLOAT_SCALAR_ARRAY, SCALAR_ARRAY_SHAPE)
      ..setTensorBufferShape(SCALAR_ARRAY_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_SCALAR_ARRAY.lossPrecision,
        /*expectedIntArr=*/ INT_SCALAR_ARRAY,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_SCALAR_ARRAY_CAPPED,
        /*expectedIntArr=*/ INT_SCALAR_ARRAY_CAPPED,
      );
    builder.build().run();
  });

  test('testLoadAndGetIntArrayWithFixedSize', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(INT_ARRAY1, ARRAY1_SHAPE)
      ..setTensorBufferShape(ARRAY1_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_ARRAY1_ROUNDED,
        /*expectedIntArr=*/ INT_ARRAY1,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_ARRAY1_CAPPED,
        /*expectedIntArr=*/ INT_ARRAY1_CAPPED,
      );
    builder.build().run();
  });

  test('testLoadAndGetFloatArrayWithFixedSize', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(FLOAT_ARRAY1, ARRAY1_SHAPE)
      ..setTensorBufferShape(ARRAY1_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_ARRAY1.lossPrecision,
        /*expectedIntArr=*/ INT_ARRAY1,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_ARRAY1_CAPPED,
        /*expectedIntArr=*/ INT_ARRAY1_CAPPED,
      );
    builder.build().run();
  });

  test('testRepeatedLoadAndGetIntArrayWithSameFixedSize', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(INT_ARRAY2, ARRAY2_SHAPE)
      ..addSrcArray(INT_ARRAY3, ARRAY3_SHAPE)
      ..setTensorBufferShape(ARRAY2_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_ARRAY3_ROUNDED,
        /*expectedIntArr=*/ INT_ARRAY3,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_ARRAY3_ROUNDED,
        /*expectedIntArr=*/ INT_ARRAY3,
      );
    builder.build().run();
  });

  test('testRepeatedLoadAndGetFloatArrayWithSameFixedSize', () {
    final ArrayTestRunnerBuilder builder = ArrayTestRunnerBuilder.newInstance()
      ..addSrcArray(FLOAT_ARRAY2, ARRAY2_SHAPE)
      ..addSrcArray(FLOAT_ARRAY3, ARRAY3_SHAPE)
      ..setTensorBufferShape(ARRAY2_SHAPE)
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteFloat32,
        /*expectedFloatArr=*/ FLOAT_ARRAY3.lossPrecision,
        /*expectedIntArr=*/ INT_ARRAY3,
      )
      ..setExpectedResults(
        /*bufferType = */
        TfLiteType.kTfLiteUInt8,
        /*expectedFloatArr=*/ FLOAT_ARRAY3_ROUNDED,
        /*expectedIntArr=*/ INT_ARRAY3,
      );
    builder.build().run();
  });

  test('testRepeatedLoadIntArrayWithDifferentFixedSize', () {
    const List<int> srcArr1 = INT_ARRAY1;
    const List<int> srcArr2 = INT_ARRAY2;
    for (int dataType in ArrayTestRunner.BUFFER_TYPE_LIST) {
      final TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(<int>[srcArr1.length], dataType);
      tensorBuffer.loadIntArray(srcArr1, <int>[srcArr1.length]);
      // Load srcArr2 which had different size as srcArr1.
      expect(() => tensorBuffer.loadIntArray(srcArr2, <int>[srcArr2.length]), throwsA(isArgumentError));
      // Assert.assertThrows(
      // IllegalArgumentException.class,
      // () -> tensorBuffer.loadArray(srcArr2, new int[] {srcArr2.length}));
    }
  });

  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
  // test('', () {});
}
