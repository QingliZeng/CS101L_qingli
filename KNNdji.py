import numpy as np
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_mat_file_v73(filename):
    with h5py.File(filename, 'r') as f:
        data_tr = np.array(f['data_tr'])
        data_te = np.array(f['data_te'])
        return data_tr, data_te

def main():
    # 加载数据
    data_tr, data_te = load_mat_file_v73('/Users/qingli/Desktop/data/pub_dataset4.mat')

    # 从数据中提取训练和测试数据
    # 分离特征和标签
    X_train = data_tr[:-1].T
    y_train = data_tr[-1].T.astype(int)

    X_test = data_te[:-1].T
    y_test = data_te[-1].T.astype(int)

    # 初始化KNN分类器
    clf = KNeighborsClassifier(n_neighbors=5)  # 这里我们使用5个近邻，但你可以根据需要调整该参数

    # 训练模型
    clf.fit(X_train, y_train)
    
    # 对测试数据进行预测
    y_pred = clf.predict(X_test)
    
    # 输出准确度和其他评估指标
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
