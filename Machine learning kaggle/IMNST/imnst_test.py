import tensorflow as tf             # 1.0版本的tensorflow才有contrib.learn
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
# optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, l2_regularization_strength=1.0)
dnn_classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[200,50, 10], n_classes=10)
dnn_classifier.fit(x_train,y_train,batch_size=50,max_steps=5000)
report = dnn_classifier.evaluate(x_train,y_train)
print('dnn_report:',report)
dnn_y_predict = dnn_classifier.predict(x_test)
dnn_submission = pd.DataFrame({'ImageId':range(1,28001),'Label':dnn_y_predict})
dnn_submission.to_csv('datasets/dnn_submission.csv',index=False)



tf.contrib.learn.DNNClassifier