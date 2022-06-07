import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import 'package:tfx/src/bindings/c/c_api.dart';

class Model {
  factory Model.fromFile(String modelPath) {
    final Pointer<Utf8> cpath = modelPath.toNativeUtf8();
    final Pointer<TfLiteModel> model = TfLiteModelCreateFromFile(cpath);
    malloc.free(cpath);
    checkArgument(model.address != nullptr.address, message: 'Unable to create model from file');
    return Model._(model);
  }

  factory Model.fromBuffer(Uint8List buffer) {
    final int size = buffer.length;
    final Pointer<Uint8> ptr = calloc<Uint8>(size);
    final Uint8List externalTypedData = ptr.asTypedList(size);
    externalTypedData.setRange(0, buffer.length, buffer);
    final Pointer<TfLiteModel> model = TfLiteModelCreate(ptr.cast(), buffer.length);
    checkArgument(model.address != nullptr.address, message: 'Unable to create model from buffer');
    return Model._(model);
  }

  Model._(Pointer<TfLiteModel> ref) : _ref = ref;

  final Pointer<TfLiteModel> _ref;
  bool _deleted = false;

  void delete() {
    checkState(!_deleted, message: 'Model already deleted.');
    TfLiteModelDelete(_ref);
    _deleted = true;
  }
}

extension ModelExtension on Model {
  Pointer<TfLiteModel> get ref => _ref;
}
