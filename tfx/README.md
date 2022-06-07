# tfx

TensorFlow Lite Flutter plugin provides an easy, flexible, and fast Dart API to integrate TFLite models in flutter apps.

## ios

```Podfile
post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)
    # Parse Issue (Xcode): Module 'TensorFlowLiteCCoreML/TensorFlowLiteCMetal' not found
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '12.0'
    end
  end
end
```
