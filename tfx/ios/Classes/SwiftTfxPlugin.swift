import Flutter
import UIKit

public class SwiftTfxPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "v7lin.github.io/tfx", binaryMessenger: registrar.messenger())
        let instance = SwiftTfxPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        if ("doesnt_matter" == call.method) {
            result(nil);
        } else {
            result(FlutterMethodNotImplemented)
        }
    }
}
