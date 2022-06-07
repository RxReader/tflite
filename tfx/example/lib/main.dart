import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:tfx/tflite.dart';
import 'package:tfx_example/classifier.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),
      theme: ThemeData.light().copyWith(
        pageTransitionsTheme: PageTransitionsTheme(
          builders: <TargetPlatform, PageTransitionsBuilder>{
            TargetPlatform.android: CupertinoPageTransitionsBuilder(),
            TargetPlatform.iOS: CupertinoPageTransitionsBuilder(),
          },
        ),
      ),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('TFLite'),
      ),
      body: ListView(
        children: <Widget>[
          ListTile(
            title: Text('init'),
            onTap: () {
              TFLite.init();
            },
          ),
          ListTile(
            title: Text('runtimeVersion'),
            onTap: () {
              if (kDebugMode) {
                print(TFLite.runtimeVersion());
              }
            },
          ),
          ListTile(
            title: Text('Text classification'),
            onTap: () {
              Navigator.of(context).push<void>(CupertinoPageRoute<void>(
                builder: (BuildContext context) => ClassifierPage(),
              ));
            },
          ),
        ],
      ),
    );
  }
}

class ClassifierPage extends StatefulWidget {
  const ClassifierPage({
    super.key,
  });

  @override
  State<StatefulWidget> createState() {
    return _ClassifierPageState();
  }
}

class _ClassifierPageState extends State<ClassifierPage> {
  late final TextEditingController _controller = TextEditingController();
  final Classifier _classifier = Classifier()..init();
  final List<Widget> _children = <Widget>[];

  @override
  void dispose() {
    _controller.dispose();
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Text classification'),
      ),
      body: ListView.builder(
        itemCount: _children.length,
        itemBuilder: (BuildContext context, int index) {
          return _children[index];
        },
      ),
      bottomNavigationBar: BottomAppBar(
        child: Padding(
          padding: MediaQuery.of(context).viewInsets,
          child: Row(
            children: <Widget>[
              Expanded(
                child: TextField(
                  decoration: InputDecoration(hintText: 'Write some text here'),
                  controller: _controller,
                ),
              ),
              TextButton(
                child: Text('Classify'),
                onPressed: () async {
                  final String text = _controller.text;
                  await _classifier.init();
                  final List<double> prediction = await _classifier.classify(text);
                  setState(() {
                    _children.add(Dismissible(
                      key: GlobalKey(),
                      onDismissed: (DismissDirection direction) {},
                      child: Card(
                        child: Container(
                          padding: EdgeInsets.all(16),
                          color: prediction[1] > prediction[0] ? Colors.lightGreen : Colors.redAccent,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: <Widget>[
                              Text(
                                'Input: $text',
                                style: TextStyle(fontSize: 16),
                              ),
                              Text('Output:'),
                              Text('   Positive: ${prediction[1]}'),
                              Text('   Negative: ${prediction[0]}'),
                            ],
                          ),
                        ),
                      ),
                    ));
                    _controller.clear();
                  });
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
