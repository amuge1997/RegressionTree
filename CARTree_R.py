import numpy as np
from RegressionTree.Node_R import Node_R

class CARTree_R:
    def __init__(self):
        self.node_root = None
        self.it_feaSum = None
        self.it_maxDeep = None
        self.it_minSample = None

    def fit(self,arr_X,arr_Y,it_minSample=1,it_maxDeep=5):
        arr_X = arr_X.T
        arr_Y = arr_Y.T
        self.it_minSample = it_minSample
        self.it_maxDeep = it_maxDeep
        self.node_root = Node_R()
        self.node_root.set_deep(1)
        self.it_feaSum = arr_X.shape[0]
        self.node(self.node_root,arr_X,arr_Y)

    def stop(self,arr_X=None,it_deep=None):
        if arr_X is not None and arr_X.shape[1] <= self.it_minSample:
            return True
        elif it_deep is not None and it_deep >= self.it_maxDeep:
            return True
        return False

    # 创建节点
    def node(self,node_parent,arr_X,arr_Y):

        # 预测值
        fl_pre = np.mean(arr_Y)
        node_parent.set_pre(fl_pre)

        if self.stop(arr_X=arr_X,it_deep=node_parent.it_deep):
            # print('样本过少')
            node_parent.set_leaf()
            return

        # 计算分割点
        # 对每个特征计算最优分割点,对每个特征的分割点求最优分割点
        it_feaIdx,fl_split = self.find_best_spilt(arr_X,arr_Y)
        if it_feaIdx is None:               # 样本无法划分,即特征相同,将该节点作为叶子节点,停止分裂
            # print('结束')
            node_parent.set_leaf()
            return
        node_parent.set_fea(it_feaIdx)
        node_parent.set_split(fl_split)

        # 左分割
        arr_lessIdx = np.where(arr_X[it_feaIdx] <= fl_split)[0]
        arr_leftX = arr_X[:,arr_lessIdx]
        arr_leftY = arr_Y[:,arr_lessIdx]
        node_left = Node_R()
        node_left.set_deep(node_parent.it_deep+1)
        node_parent.set_left(node_left)
        self.node(node_left,arr_leftX,arr_leftY)

        # 右分割
        arr_moreIdx = np.where(arr_X[it_feaIdx] > fl_split)[0]
        arr_righX = arr_X[:,arr_moreIdx]
        arr_righY = arr_Y[:,arr_moreIdx]
        node_righ = Node_R()
        node_righ.set_deep(node_parent.it_deep+1)
        node_parent.set_righ(node_righ)
        self.node(node_righ,arr_righX,arr_righY)

    # 找到最优分割特征和分割点
    def find_best_spilt(self,arr_X,arr_Y):
        # 输入 arr_X,arr_Y
        fl_minSplit = None
        it_minFea = None
        for it_feaidx in range(self.it_feaSum):
            fl_split,fl_error = self.find_best_spilt_from_fea(it_feaidx,arr_X,arr_Y)
            if it_minFea is None and fl_split is not None:
                fl_minSplit = fl_split
                it_minFea = it_feaidx
            if fl_minSplit is not None and fl_split<=fl_minSplit:
                fl_minSplit = fl_split
                it_minFea = it_feaidx

        # 返回 it_feature,fl_split
        return it_minFea,fl_minSplit

    # 从一个特征中找到最优分割点
    def find_best_spilt_from_fea(self,it_feaidx,arr_X,arr_Y):
        arr_feaX = arr_X[it_feaidx,:]
        arr_sortX = np.sort(arr_feaX)
        arr_unqiX = np.unique(arr_sortX)

        if arr_unqiX.shape[0] == 1:
            return None,None

        arr_temp1 = arr_unqiX[1:]
        arr_temp2 = arr_unqiX[:-1]
        arr_temp3 = (arr_temp1 + arr_temp2)/2
        arr_split = arr_temp3
        arr_bestSplit = np.zeros((arr_split.shape))

        for it_idx,fl_split in enumerate(arr_split):
            arr_lessYidx = np.where(arr_feaX <= fl_split)
            arr_moreYidx = np.where(arr_feaX > fl_split)

            arr_lessY = arr_Y[0,arr_lessYidx[0]]
            arr_moreY = arr_Y[0,arr_moreYidx[0]]
            fl_error = self.cal_error(arr_lessY,arr_moreY)

            arr_bestSplit[it_idx] = fl_error
        fl_minError = np.min(arr_bestSplit)
        fl_minErrorIdx = np.argmin(arr_bestSplit)
        fl_minSplit = arr_split[fl_minErrorIdx]
        # 返回 fl_split
        return fl_minSplit,fl_minError

    # 方差计算
    def cal_error(self,arr_Y1,arr_Y2):
        if arr_Y1.shape[0] == 0:
            fl_error = np.var(arr_Y2)
        elif arr_Y2.shape[0] == 0:
            fl_error = np.var(arr_Y1)
        else:
            fl_error = np.var(arr_Y1) + np.var(arr_Y2)
        return fl_error

    # 预测
    def predict(self,arr_X):
        def pre(node,arr_xt):
            if not node.bl_leaf:
                it_fea = node.it_fea
                fl_split = node.fl_split
                node_left = node.tree_left
                node_righ = node.tree_righ
                if arr_xt[it_fea] <= fl_split:
                    return pre(node_left,arr_xt)
                elif arr_xt[it_fea] > fl_split:
                    return pre(node_righ,arr_xt)
            else:
                return node.fl_pre
        it_sampleFeaSum = arr_X.shape[1]
        if it_sampleFeaSum != self.it_feaSum:
            raise Exception('CARTree_R.predict() 输入样本维度错误!')

        it_XSum = arr_X.shape[0]
        arr_Y = np.zeros((it_XSum,1))
        for it_idx in range(it_XSum):
            arr_x = arr_X[it_idx,:].reshape(-1,1)
            arr_Y[it_idx] = pre(self.node_root,arr_x)

        return arr_Y
















