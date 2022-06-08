#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint tfx.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'tfx'
  s.version          = '0.0.2'
  s.summary          = 'TensorFlow Lite Flutter plugin provides an easy, flexible, and fast Dart API to integrate TFLite models in flutter apps.'
  s.description      = <<-DESC
TensorFlow Lite Flutter plugin provides an easy, flexible, and fast Dart API to integrate TFLite models in flutter apps.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  s.static_framework = true
  s.subspec 'vendor' do |sp|
    sp.dependency 'TensorFlowLiteObjC', '2.9.1'
    sp.dependency 'TensorFlowLiteObjC/CoreML', '2.9.1'
    sp.dependency 'TensorFlowLiteObjC/Metal', '2.9.1'
  end

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
end
