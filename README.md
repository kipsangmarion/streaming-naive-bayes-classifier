# Streaming Naive Bayes Classifier

## Overview
This Python code implements a streaming version of the Naive Bayes classifier. The algorithm is designed to handle large datasets in a streaming fashion, updating its model incrementally as new data arrives. The Naive Bayes classifier is a probabilistic model based on Bayes' theorem and is commonly used for text classification, spam filtering, and other tasks.

## Problem Solved
The main goal of this algorithm is to provide a scalable and memory-efficient solution for Naive Bayes classification when dealing with large databases. It allows for the model to be updated with new instances without needing to store the entire dataset in memory. The code provides a foundation for real-time classification tasks, adapting to the continuous stream of incoming data.

## Usage
1. **Initialization:**
   - Create an instance of the `StreamingNaiveBayes` class by specifying the classes for classification.

    ```python
    classes = ['spam', 'ham']
    model = StreamingNaiveBayes(classes)
    ```

2. **Updating the Model:**
   - Update the model with new instances and their corresponding class labels using the `update_model` method.

    ```python
    instance1 = {'word1': 'hello', 'word2': 'world'}
    model.update_model(instance1, 'ham')

    instance2 = {'word1': 'discount', 'word2': 'offer'}
    model.update_model(instance2, 'spam')
    ```

3. **Updating Class Probabilities:**
   - After updating the model, use the `update_class_probabilities` method to recalculate class probabilities.

    ```python
    model.update_class_probabilities()
    ```

4. **Making Predictions:**
   - Use the `predict` method to make predictions for new instances.

    ```python
    new_instance = {'word1': 'hello', 'word2': 'world'}
    prediction = model.predict(new_instance)
    print("Predicted class:", prediction)
    ```

## Notes
- This implementation assumes feature independence, which is a naive but often effective assumption.
- The code includes basic smoothing for unseen feature values.


Feel free to customize and expand the program to better fit your specific use case.
