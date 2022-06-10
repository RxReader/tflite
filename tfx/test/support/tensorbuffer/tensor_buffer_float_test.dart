import 'package:flutter_test/flutter_test.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer_float.dart';

void main() {
  test('testCreateDynamic', () {
    final TensorBufferFloat tensorBufferFloat = TensorBufferFloat.dynamic();
    expect(tensorBufferFloat.isDynamic, isTrue);
  });

  test('testCreateFixedSize', () {
    final List<int> shape = <int>[1, 2, 3];
    final TensorBufferFloat tensorBufferFloat = TensorBufferFloat.shape(shape);
    expect(tensorBufferFloat.flatSize, 6);
  });

  test('testCreateFixedSizeWithScalarShape', () {
    final List<int> shape = <int>[];
    final TensorBufferFloat tensorBufferFloat = TensorBufferFloat.shape(shape);
    expect(tensorBufferFloat.flatSize, 1);
  });

  test('testCreateWithInvalidShape', () {
    final List<int> shape = <int>[1, -1, 2];
    expect(() => TensorBufferFloat.shape(shape), throwsA(isArgumentError));
  });

  test('testCreateUsingShapeWithZero', () {
    final List<int> shape = <int>[1, 0, 2];
    final TensorBufferFloat tensorBufferFloat = TensorBufferFloat.shape(shape);
    expect(tensorBufferFloat.flatSize, 0);
  });

  test('testGetDataType', () {
    final TensorBufferFloat tensorBufferFloat = TensorBufferFloat.dynamic();
    expect(tensorBufferFloat.dataType, TfLiteType.kTfLiteFloat32);
  });
}
