''' Traditional ml methods'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from dataset import class_list, test_path, train_path


def get_data(input_path):
    contents, labels = [], []
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            content, label = line.strip().split('\t')
            contents.append(content)
            labels.append(int(label))
    return (contents, labels)


def MNB_Classifier():
    return Pipeline([
        ('count_vec', CountVectorizer(ngram_range=(2, 3), analyzer='char_wb')),
        ('clf', MultinomialNB())
    ])


def main():
    model = MNB_Classifier()
    train_data, train_labels = get_data(train_path)
    test_data, test_labels = get_data(test_path)

    model.fit(train_data, train_labels)
    predicted = model.predict(test_data)

    report = classification_report(test_labels, predicted,
                                   target_names=class_list,
                                   digits=4, zero_division=0)
    print("Method: Naive Bayes")
    print(report)


if __name__ == "__main__":
    main()
