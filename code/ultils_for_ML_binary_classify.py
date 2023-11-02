# -*- coding: utf-8 -*-
# +
import os
import pandas as pd
import numpy as np
import copy
import umap
import matplotlib.pyplot as plt 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn import metrics
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import roc_curve, RocCurveDisplay, brier_score_loss, precision_score, recall_score, f1_score, PrecisionRecallDisplay
try:
    from sklearn.neighbors.classification import KNeighborsClassifier
except:
    from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.manifold.t_sne import TSNE
except:
    from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier
from sklearn.model_selection import RepeatedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from imblearn.over_sampling import SMOTE  # 选取少数类样本插值采样

from typing import List
import pandas
import numpy
# -

# 设置全局随机数种子
from global_var import seed, random_state
import random
random.seed(seed)
np.random.seed(seed)

from scipy.stats import uniform
class int_uniform():
    def __init__(self,loc,scale):
        self.loc = loc
        self.scale = scale
    
    def rvs(self, size=None,random_state=None):
        sample = uniform(loc=self.loc, scale=self.scale).rvs(size=size,random_state=random_state)
        return np.around(sample).astype(int)


from sklearn.utils.validation import _deprecate_positional_args
# @_deprecate_positional_args
def plot_roc_curve(
    estimator, X, y, *, sample_weight=None,
    drop_intermediate=True, response_method="auto",
    name=None, ax=None, **kwargs): 
    
    check_matplotlib_support('plot_roc_curve')

    classification_error = (
        "{} should be a binary classifier".format(estimator.__class__.__name__)
    )
    if not is_classifier(estimator):
        raise ValueError(classification_error)
    
    y_pred = estimator.predict_proba(X)

    if y_pred.ndim != 1:
        if y_pred.shape[1] != 2:
            raise ValueError(classification_error)
        else:
            y_pred = y_pred[:, 1]

    pos_label = estimator.classes_[1]
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)
    name = estimator.__class__.__name__ if name is None else name
    viz = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name
    )
    viz.threshold = threshold
    return viz.plot(ax=ax, name=name, **kwargs)


# 画混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(
    cm,target_names,
    title='Confusion Matrix',
    fn='Confusion Matrix.png',
    fmt='.2g',
    center=None,
    dpi=100,
    show=False,
):
    '''
    Example:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true = truelabel, y_pred = predClasses)
    sns.set()
    plot_cm(cm,target_names,
            title='Confusion Matrix Model',
            fn='Confusion Matrix Model.png',
            fmt='.20g',
            center=cm.sum()/num_classes
           )
    '''
    f,ax = plt.subplots(figsize=(5,3),dpi=dpi)
    ax = sns.heatmap(cm,annot=True,fmt=fmt,center=center,annot_kws={'size':20,'ha':'center','va':'center'})#fmt='.20g',center=250
    ax.set_title(title,fontsize=20)#图片标题文本和字体大小
    ax.set_xlabel('Predict',fontsize=20)#x轴label的文本和字体大小
    ax.set_ylabel('Ground-Truth',fontsize=20)#y轴label的文本和字体大小
    ax.set_xticklabels(target_names,fontsize=20)#x轴刻度的文本和字体大小
    ax.set_yticklabels(target_names,fontsize=20)#y轴刻度的文本和字体大小
    #设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    
    if fn:
        plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return f,ax

def plot_3cm(cm,target_names,num_classes,clf_name,dir_result,show=False):
    plot_confusion_matrix(cm,target_names,
                          title='Confusion Matrix\n'+clf_name,
                          fn=os.path.join(dir_result,'Confusion Matrix '+clf_name+'.png'),
                          fmt='.20g',
                          center=cm.sum()/num_classes,
                          show=show
                         )
    
    cm_norm_recall = cm.astype('float') / cm.sum(axis=0) 
    cm_norm_recall = np.around(cm_norm_recall, decimals=3)
    plot_confusion_matrix(cm_norm_recall,target_names,
                          title='Row-Normalized Confusion Matrix\n'+clf_name,
                          fn=os.path.join(dir_result,'Row-Normalized Confusion Matrix '+clf_name+'.png'),
                          fmt='.3g',
                          center=0.5,
                          show=show,
                         )
    
    cm_norm_precision = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    cm_norm_precision = np.around(cm_norm_precision, decimals=3)
    plot_confusion_matrix(cm_norm_precision,target_names,
                          title='Column-Normalized Confusion Matrix\n'+clf_name,
                          fn=os.path.join(dir_result,'Column-Normalized Confusion Matrix '+clf_name+'.png'),
                          fmt='.3g',
                          center=0.5,
                          show=show,
                         )


def visualize_data_reduced_dimension(data,
                                     reducer: str = 'UMAP',
                                     n_dim: str = 3,
                                     title: str = 'Reduced Dimension Projection',
                                     dir_result: str = './',
                                     show: bool = False):
    '''
    数据降维可视化
        data:字典，包含：
            X：输入特征
            y:真实类别
    '''
    assert reducer in ['UMAP','TNSE']
    target_names = data['target_names']#类别名称
    if reducer == 'TNSE':
        reducer = TSNE(n_components = n_dim, random_state=42)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=n_dim, random_state=42)
        X_embedding = reducer.fit_transform(data['X']) 
        
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    if n_dim==3:
        X2d_zmin,X2d_zmax = np.min(X_embedding[:,2]), np.max(X_embedding[:,2])
        
    #plot 
    fig = plt.figure(dpi=100)#figsize=(8,6))
    colors = ['red','green','blue','orange','cyan']
    markers = ['o','s','^','*']
    if n_dim == 2:
        for i,target_name in enumerate(target_names):
            idx = np.where(data['y']==i)[0].tolist()#根据真实类别标签来绘制散点
            plt.scatter(X_embedding[idx, 0].squeeze(), 
                        X_embedding[idx, 1].squeeze(), 
                        c=colors[i],# cmap='Spectral', 
                        marker=markers[i],
                        s=10,
                        alpha=0.5, 
                        edgecolors='none',
                        label=target_name,
                       )
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')  
    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
#         plt.zlim(X2d_zmin,X2d_zmax)
        for i,target_name in enumerate(target_names):
            idx = np.where(data['y']==i)[0].tolist()#根据真实类别标签来绘制散点
            ax.scatter(X_embedding[idx, 0].squeeze(), 
                       X_embedding[idx, 1].squeeze(), 
                       X_embedding[idx, 2].squeeze(),
                       c=colors[i],
                       s=10,
                       marker=markers[i],
                       alpha=0.5,
                       label=target_name,
                      )
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2') 
        ax.set_zlabel('dim 3')
#     plt.xlim(X2d_xmin,X2d_xmax)
#     plt.ylim(X2d_ymin,X2d_ymax)       
    plt.legend(labels=target_names)
    plt.title(title)
    plt.tight_layout()
    fn = os.path.join(dir_result,title.replace('/n','_')+".png")
    plt.savefig( fn )
    if show:
        plt.show()
    else:
        plt.close()
    return 


def visualize_reduced_decision_boundary(clf,
                                        data,
                                        reducer: str = 'UMAP',
                                        title: str = "Decision-Boundary of the Trained Classifier in Reduced-Features-Space",
                                        dir_result: str = './',
                                        show: bool = False):
    '''
    分类器的决策边界可视化
    clf:分类器
    X：输入特征
    y:真实类别
    '''
    assert reducer in ['TNSE','UMAP']
    if reducer == 'TNSE':
        reducer = TSNE(n_components = 2, random_state=42)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_embedding = reducer.fit_transform(data['X']) 
    
    y_predicted = clf.predict(data['X'])#根据y_predicted 结合KNeighborsClassifier来确定决策边界
    
    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    # 创建meshgrid 
    resolution = 400 #100x100背景像素
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    xx,yy = np.meshgrid(np.linspace(X2d_xmin,X2d_xmax,resolution), np.linspace(X2d_ymin,X2d_ymax,resolution))
     
    #使用1-NN 
    #在分辨率x分辨率网格上近似Voronoi镶嵌化
    background_model = KNeighborsClassifier(n_neighbors = 1).fit(X_embedding,y_predicted)
    voronoiBackground = background_model.predict( np.c_[xx.ravel(),yy.ravel()] )
    voronoiBackground = voronoiBackground.reshape((resolution,resolution))
     
    #plot 
    plt.figure(dpi=100)#figsize=(8,6))
    plt.contourf(xx,yy,voronoiBackground,alpha=0.2)
    idx_0 = np.where(data['y']==0)[0].tolist()#根据真实类别标签来绘制散点
    idx_1 = np.where(data['y']==1)[0].tolist()#根据真实类别标签来绘制散点
    plt.scatter(X_embedding[idx_0, 0], X_embedding[idx_0, 1], 
                c='blue',# cmap='Spectral', 
                marker='o',s=20,
                label=data['classname'][0],
                )
    plt.scatter(X_embedding[idx_1, 0], X_embedding[idx_1, 1], 
                c='orange',# cmap='Spectral', 
                marker='s',s=20,
                label=data['classname'][1],
                )
    plt.legend(labels=data['classname'])
    plt.xlim(X2d_xmin,X2d_xmax)
    plt.ylim(X2d_ymin,X2d_ymax)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(title)
    plt.savefig( os.path.join(dir_result,title.replace('/n','_')+".png") )
    if show:
        plt.show()
    else:
        plt.close()
    return


def get_DCA_coords(y_true:numpy.ndarray, y_proba:numpy.ndarray, eps=1e-6):
    """
    DCA曲线
    y_proba: shape是(num_samples,)
    y_true: shape是(num_samples,)
    """
    if type(y_true)==pandas.core.frame.DataFrame:
        y_true = y_true.values
    y_true = y_true.squeeze()
    N = len(y_true)# 总数据量
    
    pt_arr = []
    net_bnf_arr = []
    y_proba = y_proba.ravel()
    for pt in np.arange(0,1,0.01):
        #compute TP FP
        y_proba_clip = (y_proba>pt).astype(int)
        TP = np.sum( y_true*np.round(y_proba_clip) )
        FP = np.sum((1 - y_true) * np.round(y_proba_clip))
        net_bnf = ( TP-(FP * (pt+eps)/(1-pt+eps)) )/N
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    # all negtive
    allneg_net_bnf_arr = np.zeros_like(pt_arr)
    # all positive
    pt_np = np.array(pt_arr)
    pi = np.sum(y_true)/len(y_true)# 患病率
    allpos_net_bnf_arr = pi-(1-pi)*pt_np/(1-pt_np)
    return pt_arr, net_bnf_arr, allpos_net_bnf_arr, allneg_net_bnf_arr


def plot_DCA(y_proba:numpy.ndarray, y_true:numpy.ndarray, clf_name='', fn:str="DCA.png", show=False):
    """
    DCA曲线
    y_proba: shape是(num_samples,)
    y_true: shape是(num_samples,)
    """
    if type(y_true)==pandas.core.frame.DataFrame:
        y_true = y_true.values
    y_true = y_true.squeeze()
    N = len(y_true)# 总数据量
    
    pt_arr = []
    net_bnf_arr = []
    y_proba = y_proba.ravel()
    for pt in np.arange(0,y_proba.max(),0.01):
        #compute TP FP
        y_proba_clip = (y_proba>pt).astype(int)
        TP = np.sum( y_true*np.round(y_proba_clip) )
        FP = np.sum((1 - y_true) * np.round(y_proba_clip))
        net_bnf = ( TP-(FP * pt/(1-pt)) )/N
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    fig = plt.figure(figsize=(4,4),dpi=300)
    plt.plot(pt_arr, net_bnf_arr, color='red', lw=2, linestyle='-',label=clf_name)
    plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All Negative')
    pt_np = np.array(pt_arr)
    pi = np.sum(y_true)/len(y_true)# 患病率
    all_pos = pi-(1-pi)*pt_np/(1-pt_np)
    plt.plot(pt_arr, all_pos , color='b', lw=1, linestyle='dotted',label='All Positive')
    plt.xlim([0.0, 1.0])
    plt.ylim([max(-0.15,min(net_bnf_arr)*1.2), max(0.4,max(net_bnf_arr)*1.2)])
    plt.ylim([-0.15, max(0.4,max(net_bnf_arr)*1.2)])
    plt.xlabel('Probability Threshold')
    plt.ylabel('Net Benefit')
    plt.title('DCA')
    plt.legend(labels=['All Negative','All Positive'])
    plt.grid("on")
    plt.savefig(fn)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

# 1. Filter方法(逐个特征分析，没有考虑到特征之间的关联作用，可能把有用的关联特征误踢掉。)
#     1.1 移除低方差的特征 (Removing features with low variance)
#     1.2 单变量特征选择 (Univariate feature selection)
#         1.2.1 卡方(Chi2)检验
#         1.2.2 互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)
#         1.2.3 基于模型的特征排序 (Model based ranking)
# 2. Wrapper
#     2.1 RFE
# 3.Embedding
#     3.1 使用SelectFromModel选择特征 (Feature selection using SelectFromModel)
#         3.1.1 基于L1的特征选择 (L1-based feature selection)


# +
def plt_feature_importance(feat_imp,dir_result='./',fig=None):
    # 画特征重要性bar图
    plt.figure(dpi=300)
    y_pos = np.arange(feat_imp.shape[0])
    plt.barh(y_pos, feat_imp.values.squeeze(), align='center', alpha=0.8)
    plt.yticks(y_pos, feat_imp.index.values)
    plt.xlabel('Feature Importance')
    fn = os.path.join(dir_result,'Feature Importance.png')
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return fig

# Filter
def select_features_Chi2(X,y,kbest=10,dir_result='./'):
    """
    采用卡方检验(Chi2)方法(SelectKBest)选择特征
    注意：经典的卡方检验是检验定性自变量对定性因变量的相关性。
    注意：Input X must be non-negative.
    """
    from sklearn.feature_selection import SelectKBest, chi2
    fit = SelectKBest(score_func=chi2, k=kbest).fit(X, y)
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    return index_selected_features

def selelct_features_MIC(X,y,kbest=15,dir_result='./'):
    """采用互信息指标，来筛选特征"""
    from minepy import MINE#minepy包——基于最大信息的非参数估计
    m = MINE()
    # 单独采用每个特征进行建模
    feature_names = X.columns
    importance = []
    import pandas
    if type(y)==pandas.core.frame.DataFrame:
        y = y.values.squeeze() 
    else:
        y = y.squeeze() 
    for i in range(X.shape[1]):
        m.compute_score( X.iloc[:, i], y)
        importance.append( round(m.mic(),3) )
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    feat_imp = feat_imp.iloc[:kbest]
    
    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)

    return feat_imp

# Wrapper
def select_features_RFE(X,y,base_model,kbest=15,dir_result='./'):
    """采用RFE方法选择特征，用户可以指定base_model"""
    from sklearn.feature_selection import RFE
    rfe = RFE(base_model, n_features_to_select=kbest)
    fit = rfe.fit(X, y)
    
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    importance = fit.estimator_.coef_.squeeze().tolist()
    #importance = [fit.ranking_[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    
    # 画图
    fig = plt.figure(dpi=100)#,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
        
    return feat_imp

# Embedding
def select_features_LSVC(X,y,max_features:int=15,dir_result='./'):
    """采用LSVC方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC

    my_model = LinearSVC(C=0.01, penalty="l1", dual=False).fit( X, y )
    selector = SelectFromModel(my_model,prefit=True,max_features=max_features)
    importance = selector.estimator.coef_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    
    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    
    return feat_imp   

def select_features_LR(X,y,max_features:int=15,dir_result='./'):
    """采用带L1和L2惩罚项的逻辑回归作为基模型的特征选择,
    参数threshold为权值系数之差的阈值""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression as LR

    my_model = LR(C=0.1).fit( X, y )
    selector = SelectFromModel(my_model,prefit=True,max_features=max_features)
    importance = selector.estimator.coef_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
        
    return feat_imp

def select_features_Tree(X,y,max_features:int=15,dir_result='./'):
    """采用Tree的方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
#     selector = SelectFromModel(ExtraTreesClassifier()).fit(X, y)
#     index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    
    my_model = ExtraTreesClassifier().fit( X, y )
    selector = SelectFromModel(my_model,prefit=True,max_features=max_features)
    importance = selector.estimator.feature_importances_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    
    return feat_imp

def select_features_RF(X,y,max_features:int=15,dir_result='./'):
    """基于模型（此处采用随机森林交叉验证）的特征排序，"""
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    RF = RandomForestRegressor(n_estimators=20, max_depth=4)
    
    my_model = RF.fit( X, y )
    selector = SelectFromModel(my_model,prefit=True,max_features=max_features)
    importance = selector.estimator.feature_importances_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    return feat_imp

# def select_features_mrmr(X,method='MID',kbest=10, dir_result='./'):
#     """
#     采用mRMR方法筛选特征(该方法不考虑应变量)
#     X是dataframe
#     """
#     import pymrmr
#     name_selected_features = pymrmr.mRMR(X, method, kbest)#也可以输入dataframe
#     feat_imp = pd.DataFrame( data=np.zeros([kbest,1]), index=name_selected_features, columns=['feature importance'])
#     return feat_imp


# -

def select_features(X,y,method,dir_result,**kwargs):
    """多种特征选择方法的封装函数"""
    if 'kbest' in kwargs.keys():
        kbest = kwargs['kbest']
    else:
        kbest = 15

    if method == 'MIC':
        feat_imp = selelct_features_MIC(X,y,kbest=kbest,dir_result=dir_result)
    elif method == 'RFE':
        from sklearn.linear_model import LogisticRegression as LR
        LogR = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
           intercept_scaling=1,class_weight=None,random_state=None,
           solver='liblinear',max_iter=1000,multi_class='ovr',
           verbose=0,warm_start=False,n_jobs=1)
        feat_imp = select_features_RFE(X,y,base_model=LogR,kbest=kbest,dir_result=dir_result)
    elif method == 'EmbeddingLSVC':
        feat_imp = select_features_LSVC(X,y,max_features=kbest,dir_result=dir_result)
    elif method == 'EmbeddingLR':
        feat_imp = select_features_LR(X,y,max_features=kbest,dir_result=dir_result)
    elif method == 'EmbeddingTree':
        feat_imp = select_features_Tree(X,y,max_features=kbest,dir_result=dir_result)
    elif method == 'EmbeddingRF':
        feat_imp = select_features_RF(X,y,max_features=kbest,dir_result=dir_result)
    elif method == 'mRMR':
        feat_imp = select_features_mrmr(X,'MID',kbest=kbest,dir_result=dir_result)

    Data = {'X':X.loc[:,feat_imp.index],'y':y}
    return Data,feat_imp



# +
def boxplot(x, x_names, title='AUC of different Algorithms', fn_save='AUC-of-different-Algorithms.png', show=True):
    # 参考：https://blog.csdn.net/roguesir/article/details/78249864 
    fig=plt.figure(dpi=100)
    plt.boxplot(x,patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值 
                boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色           
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色 
                medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
    plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    plt.ylabel('Cross Validation AUC')
    plt.title(title)
    plt.savefig(fn_save, bbox_inches='tight')
    if show:
        plt.show() 
    else:
        plt.close()
    return

def violinplot(x,x_names,title='AUC of different Algorithms',fn_save='AUC-of-different-Algorithms.png',show=True):
    # 参考：https://blog.csdn.net/roguesir/article/details/78249864 
    fig = plt.figure(dpi=300)
    plt.violinplot(
        x,
        showmeans=True, 
        showmedians=True,
        showextrema=True) # 设置中位数线的属性，线的类型和颜色
    plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    plt.ylabel('Cross Validation AUC')
    plt.title(title)
    plt.savefig(fn_save, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return


# -

def get_algorithm_test_result(
    X,y,
    target_names:List[str],
    classifier_fitted,
    clf_name:str=None,
    dataset_name:str=None,
    dir_result:str='./',
    show=False,
):
    """
    外部验证，输出roc曲线、混淆矩阵
    classifier_fitted： 拟合好的分类器。
    """
    assert type(X) in [pandas.core.frame.DataFrame, numpy.ndarray]
    assert type(y) in [pandas.core.frame.DataFrame, numpy.ndarray]
    if type(X)==pandas.core.frame.DataFrame:
        X = X.values
    if type(y)==pandas.core.frame.DataFrame:
        y = y.values.squeeze()
    
    dir_result = os.path.join(dir_result,dataset_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
    
    # (1)ROC曲线及其AUC
    fig, ax = plt.subplots(figsize=(4,4),dpi=300)
    title = f'ROC Curve of {clf_name} on {dataset_name}'
    fn = os.path.join(dir_result,title.replace('\n','_'))
    viz = plot_roc_curve(classifier_fitted, X, y, label=None, name=None, ax=ax)
    roc_auc = viz.roc_auc
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend([f'AUC = {roc_auc:.3f}','Chance'],loc="lower right")
    plt.savefig( fn )
    if show:
        plt.show()
    else:
        plt.close()
    
    # (2)PRC曲线
    fig, ax = plt.subplots(figsize=(4,4),dpi=300)
    PrecisionRecallDisplay.from_estimator( classifier_fitted, X, y, name=clf_name, ax=ax )#, plot_chance_level
    title = f'Precision-Recall Curve of {clf_name} on {dataset_name}'
    ax.set_title(title)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    y_score = classifier_fitted.predict_proba(X)
    pr_auc = average_precision_score(y, y_score[:,1])
    plt.legend(labels=[f'{clf_name} (PR-AUC = {pr_auc:.3f})'], loc='lower left',markerfirst=True)
    fn = os.path.join(dir_result,title.replace('\n','_'))
    plt.savefig(fn)
    if show:
        plt.show()
    else:
        plt.close()

        
    # (3)画混淆矩阵
    y_pred = classifier_fitted.predict(X)
    cm = confusion_matrix( y_true = y, y_pred = y_pred )
    num_classes = len(target_names)
    plot_3cm(cm,target_names,num_classes,clf_name,dir_result,show=show)
    

    # (4)classification_reportreport
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true=y, 
        y_pred=y_pred, 
        labels=sorted(np.unique(y).tolist()),
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    df_report = pd.DataFrame(report)
    ## 补充roc_auc
    df_roc_auc = pd.DataFrame(roc_auc,index=['ROC-AUC'],columns=df_report.columns)# 
    ## 补充pr_auc
    df_pr_auc = pd.DataFrame(pr_auc,index=['PR-AUC'],columns=df_report.columns)# 
    df_report = pd.concat([df_roc_auc, df_pr_auc, df_report], axis=0).round(decimals=3)
    ## 转置
    df_report = df_report.T
    df_report['support'] = df_report['support'].astype(int)
    # accuracy
    df_report.loc['accuracy',:] = df_report.loc['accuracy','recall']
    ## 保存
    fn = os.path.join(dir_result,'classification report of '+clf_name+' on test set.csv')
    df_report.to_csv(fn)
    display(df_report)
    
    # (5)DCA
    fn = os.path.join(dir_result,"DCA.png")
    plot_DCA(y_proba=classifier_fitted.predict_proba(X)[:,1].squeeze(), y_true=y.squeeze(), clf_name=clf_name, fn=fn, show=show)
    
    # (6)记录
    ret = dict(roc_auc=roc_auc, pr_auc=pr_auc, cm=cm, report=df_report)
    return ret


# +

def plot_calibration_curve_binary_class(
    clf, 
    clf_name, 
    X_train, y_train,
    X_test, y_test,
    cv='prefit',#5
    n_bins:int=10,
    dir_result:str='./',
    show=False):
    """Plot calibration curve for est w/o and with calibration. """
    
    # Calibrated with sigmoid calibration
    calibrated_classifier = CalibratedClassifierCV(clf, cv=cv, method='sigmoid')

    fig = plt.figure(dpi=300,figsize=(5, 5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for est, name in [(clf, clf_name),
                      (calibrated_classifier, "calibrated "+clf_name)]:
        est.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(est, "predict_proba"):
            prob_pos = est.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = est.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        est_score = brier_score_loss(y_test, prob_pos)

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=n_bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, est_score))

        ax2.hist(prob_pos, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives",fontsize=10)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)',fontsize=10)

    ax2.set_xlabel("Mean predicted value",fontsize=10)
    ax2.set_ylabel("Count",fontsize=10)
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    
    fn = os.path.join(dir_result,'Calibration plots.png')
    plt.savefig( fn )
    if show:
        plt.show()
    else:
        plt.close()
    return fig,calibrated_classifier


# -

def save_text(filename, contents):
    fh = open(filename, 'w', encoding='utf-8')
    fh.write(contents)
    fh.close()    

# +
def get_algorithm_result(
    X_train,
    y_train,
    X_test,
    y_test,
    X_external,
    y_external,
    target_names,
    classifier,
    clf_name, 
    dir_result='./'
):
    """
    运行单一算法，给出各种结果:
        内部训练集上:
            roc曲线
            pr曲线
            混淆矩阵（３个）
        内部测试集上：
            roc曲线
            pr曲线
            混淆矩阵（３个）
        外部数据集上：
            roc曲线
            pr曲线
            混淆矩阵（３个）
    """
    ret = {}
    dir_result = os.path.join(dir_result,clf_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
        
    if type(X_train)==pandas.core.frame.DataFrame:
        X_train = X_train.values
    if  type(y_train)==pandas.core.frame.DataFrame:
        y_train = y_train.values.squeeeze()
    if type(X_test)==pandas.core.frame.DataFrame:
        X_test = X_test.values
    if  type(y_test)==pandas.core.frame.DataFrame:
        y_test = y_test.values.squeeze()   

    # 检查模型是否已经拟合了训练数据集
    if hasattr(classifier, 'coef_') and hasattr(classifier, 'intercept_'):
        print("The model has been fit on the training dataset.")
    else:
        print("The model has not been fit on the training dataset.")
        classifier.fit(X_train,y_train)
        
    # 2、获取训练集的结果
    ret_train = get_algorithm_test_result(
        X = X_train,
        y = y_train,
        target_names = target_names,
        classifier_fitted = classifier,
        clf_name = clf_name,
        dataset_name = 'Train Set',
        dir_result = dir_result) 
    
    # 3、获取测试集上的roc,auc,混淆矩阵,report
    ret_test = get_algorithm_test_result(
        X = X_test,
        y = y_test,
        target_names = target_names,
        classifier_fitted = classifier,
        clf_name = clf_name,
        dataset_name = 'Test Set',
        dir_result = dir_result) 
    
    # 4、外部数据集的结果
    if X_external and y_external:
        ret_external = get_algorithm_test_result(
            X = X_external,
            y = y_external,
            target_names = target_names,
            classifier_fitted = classifier,
            clf_name = clf_name,
            dataset_name = 'External Test Set',
            dir_result = dir_result
        )
        
#     # 5、画决策边界（内部训练、测试集）
#     visualize_reduced_decision_boundary(
#         clf=classifier,data=Data,
#         title=f"Decision-Boundary of {clf_name}",
#         dir_result=dir_result
#     ) 
    
    # 6、校正曲线 
    fig,calibrated_classifier = plot_calibration_curve_binary_class(
        clf = classifier, 
        clf_name = clf_name, 
        X_train = X_train, 
        y_train = y_train,
        X_test = X_test, 
        y_test = y_test,
        cv = 'prefit',
        n_bins = 10,
        dir_result = os.path.join(dir_result,'Test Set'),
    )
    
    #　结果变量保存
    ret['clf_name'] = clf_name
    ret['classifier'] = classifier
    ret['train'] = ret_train
    ret['test'] = ret_test
    if X_external and y_external:
        ret['external_test'] = ret_external

    return ret


# -

def run_algorithms(data_internal,
                   data_external=None,
                   clf_names:list=None,
                   feature_select_method:str='RFE',
                   kbest:int=15,
                   dir_result='./'):
    """运行多种算法"""
    
    target_names = data_internal['target_names']
    
    # SMOTE数据的扩增以缓和类别不均衡的影像
    smo = SMOTE(sampling_strategy='auto',random_state=random_state)# 实例化
    X_train, y_train = smo.fit_resample(data_internal['X_train'].values, data_internal['y_train'].values)
    X_train = pd.DataFrame(data=X_train,columns=data_internal['X_train'].columns)
    y_train = pd.DataFrame(data=y_train,columns=data_internal['y_train'].columns)
    

    # 特征选择
    Data,feat_imp = select_features(X=X_train,
                                    y=y_train,
                                    method=feature_select_method,
                                    dir_result=dir_result,
                                    kbest=kbest)
    if type(feat_imp) == list:
        selected_features = feat_imp[0].index.values.tolist()
    elif type(feat_imp) == pd.core.frame.DataFrame:
        selected_features = feat_imp.index.values.tolist()

    # 获取数据，切分数据集
    X_train, y_train = data_internal['X_train'][selected_features], data_internal['y_train']
    X_test, y_test = data_internal['X_test'][selected_features], data_internal['y_test']
    if data_external:
        X_external = data_external['X'][selected_features], 
        y_external = data_external['y']
    else:
        X_external, y_external = None, None
        

    ##　数据降维可视化
    visualize_data_reduced_dimension(
        data={'X':pd.concat([X_train,X_test],axis=0), 
              'y':np.concatenate((y_train,y_test),axis=0),
              'target_names':target_names
             },
        n_dim=2,
        title="Reduced Dimension Projection of Data selected by "+feature_select_method,
        dir_result=dir_result
    ) 
    if X_external and y_external:
        visualize_data_reduced_dimension(
            data={'X':X_external, 
                  'y':y_external, 
                  'target_names':target_names
                 },
            n_dim=3,
            title="Reduced Dimension Projection of External Data selected by "+feature_select_method,
            dir_result=dir_result
        )
    
    # 转为numpy
    X_train = X_train.values
    y_train = y_train.values.squeeze()
    X_test = X_test.values
    y_test = y_test.values.squeeze()
    if X_external and y_external:
        X_external = X_external.values
        y_external = y_external.values.squeeze()
    
    # ================================================模型训练和验证=======================================================
    my_scorer = "roc_auc_ovr"# "f1_macro" #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
    result_dict = {'selected_features': selected_features}
    for clf_name in clf_names:
        print(f'Running {clf_name}...')
        # Logistic回归
        if clf_name == 'Logistic':
            from sklearn.linear_model import LogisticRegression as LR
            Logistic_clf = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
               intercept_scaling=1, random_state=None,
               solver='liblinear',max_iter=100,#multi_class='ovr',
               verbose=0,warm_start=False,n_jobs=1)# solver='newton-cg'  'liblinear'  'lbfgs'  'sag' 'saga'
            param_grid ={ 'class_weight' : ['balanced', None] }
            search = GridSearchCV(estimator=Logistic_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            Logistic_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier = Logistic_clf,
                clf_name = 'Logistic',
                dir_result=dir_result)
            ret['search'] = search
            Logistic_clf_best = ret['classifier']
            result_dict['Logistic'] = ret
            

        # LDA==========================
        if clf_name == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            LDA_clf = LinearDiscriminantAnalysis()#solver='eigen')#, covariance_estimator=OAS() )
            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
            }
            search = GridSearchCV(estimator=LDA_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            LDA_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier = LDA_clf,
                clf_name = 'LDA',
                dir_result=dir_result)
            ret['search'] = search
            LDA_clf_best = ret['classifier']
            result_dict['LDA'] = ret
    
        # SVM==========================
        if clf_name == 'SVM':
            from sklearn.svm import SVC
            SVM_clf = SVC(decision_function_shape='ovr',probability=True)
            param_grid ={ 'kernel' : ['rbf', 'sigmoid'] }#'linear','rbf', 'poly', 'sigmoid'
            search = GridSearchCV(estimator=SVM_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            SVM_clf_best = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=SVM_clf_best, 
                clf_name='SVM',
                dir_result=dir_result)
            ret['search'] = search
            SVM_clf_best = ret['classifier']
            result_dict['SVM'] = ret
    
    
        ## KNN分类器==========================
        if clf_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            KNN_clf = KNeighborsClassifier(metric="minkowski",n_jobs=-1)
            param_grid = {
                'n_neighbors': [2,5,10],
                'weights': ['uniform', 'distance'],
                'p': [1,2],
            }
            search = GridSearchCV(estimator=KNN_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            KNN_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=KNN_clf, 
                clf_name='KNN',
                dir_result=dir_result)
            ret['search'] = search
            KNN_clf_best = ret['classifier']
            result_dict['KNN'] = ret
    
    
        # GaussianNB型朴素贝叶斯分类器
        if clf_name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            GaussianNB_clf = GaussianNB()
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=GaussianNB_clf, 
                clf_name='GaussianNB',
                dir_result=dir_result)
            ret['search'] = search
            GaussianNB_clf = ret['classifier']
            result_dict['GaussianNB'] = ret


        # 决策树==========================
        if clf_name == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            Tree_clf = DecisionTreeClassifier()
            param_grid = {'max_depth': [5, 10, 20],
                          'min_samples_leaf': [2,4,8,16],
                          'min_samples_split': [2,4,8,16],
                          'class_weight' : ["balanced",None]
                         }
            search = GridSearchCV(estimator=Tree_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            Tree_clf_best = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=Tree_clf_best, 
                clf_name='DecisionTree',
                dir_result=dir_result)
            ret['search'] = search
            Tree_clf_best = ret['classifier']
            result_dict['DecisionTree'] = ret


        # ExtraTrees========================
        if clf_name == 'ExtraTrees':
            from sklearn.ensemble import ExtraTreesClassifier
            ExtraTrees_clf = ExtraTreesClassifier()
            param_grid = {
                'n_estimators': [5, 10, 20, 40],#[10,50,100,200]
            }
            search = GridSearchCV(estimator=ExtraTrees_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            ExtraTrees_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=ExtraTrees_clf, 
                clf_name='ExtraTrees',
                dir_result=dir_result)
            ret['search'] = search
            ExtraTrees_clf = ret['classifier']
            result_dict['ExtraTrees'] = ret


        ## 随机森林========================
        if clf_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            RandomForest_clf = RandomForestClassifier(bootstrap=True, 
                                                      class_weight=None, 
                                                      criterion='gini',
                                                      min_impurity_decrease=0.0,
                                                      min_weight_fraction_leaf=0.0, 
                                                      n_jobs=-1,
                                                      oob_score=False, 
                                                      random_state=0, 
                                                      verbose=0, 
                                                      warm_start=False)
            param_grid = dict(max_depth = int_uniform(loc=5, scale=10),
                              min_samples_leaf = uniform(loc=0, scale=0.1),
                              min_samples_split = uniform(loc=0, scale=0.1),
                              n_estimators = [5, 10, 20, 40],#[10,50,100,200],#[10, 20, 35, 50]
                             )
            search = RandomizedSearchCV(estimator=RandomForest_clf, 
                                        param_distributions=param_grid,
                                        n_iter=100,
                                        n_jobs=-1,
                                        scoring='roc_auc',
                                        cv=5,
                                        refit=True)
            search = search.fit(X_train, y_train)
            RandomForest_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=RandomForest_clf, 
                clf_name= 'RandomForest',
                dir_result=dir_result
            )
            ret['search'] = search
            RandomForest_clf = ret['classifier']
            result_dict['RandomForest'] = ret


        # Bagging========================
        if clf_name == 'Bagging':
            from sklearn.ensemble import BaggingClassifier
            Bagging_clf = BaggingClassifier()
            param_grid = {'estimator': [Logistic_clf_best,
                                        LDA_clf_best,
                                        SVM_clf_best,
                                        KNN_clf_best,
                                        GaussianNB_clf,
                                        Tree_clf_best],
                          'n_estimators' : [5, 10, 20, 40],#[10,50,100,200]
                         }
            clf = GridSearchCV(estimator=Bagging_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search = clf.fit(X_train, y_train)
            Bagging_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=Bagging_clf, 
                clf_name='Bagging',
                dir_result=dir_result)
            ret['search'] = search
            Bagging_clf_best = ret['classifier']
            result_dict['Bagging'] = ret


        ## Adaboost算法========================
        if clf_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier
            AdaBoost_clf = AdaBoostClassifier(algorithm='SAMME.R') #base_estimator默认是决策树（深度为1），可修改
            param_dict = dict(estimator = [Logistic_clf_best,
                                           LDA_clf_best,
                                           SVM_clf_best,
                                           KNN_clf_best,
                                           GaussianNB_clf,
                                           Tree_clf_best],
                              n_estimators = [5, 10, 20, 40],#[10,50,100,200],
                             )
            search = GridSearchCV(estimator=AdaBoost_clf, param_grid=param_dict, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            AdaBoost_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=AdaBoost_clf, 
                clf_name='AdaBoost',
                dir_result=dir_result)
            ret['search'] = search
            AdaBoost_clf = ret['classifier']
            result_dict['AdaBoost'] = ret


        # GradientBoost========================
        if clf_name == 'GradientBoost':
            from sklearn.ensemble import GradientBoostingClassifier
            GradientBoost_clf = GradientBoostingClassifier()   
            param_dict = dict(n_estimators = [5, 10, 20, 40],#[10,50,100,200],
                              max_depth = int_uniform(loc=2, scale=10),
                             )
            search = RandomizedSearchCV(estimator=GradientBoost_clf, 
                                     param_distributions=param_dict, 
                                     n_iter=100,
                                     n_jobs=-1,
                                     scoring='roc_auc', 
                                     cv=5,
                                     refit=True)
            search.fit(X_train, y_train)
            GradientBoost_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=GradientBoost_clf, 
                clf_name='GradientBoost',
                dir_result=dir_result
            )
            ret['search'] = search
            GradientBoost_clf = ret['classifier']
            result_dict['GradientBoost'] = ret


        # MLP========================
        if clf_name == 'MLP':
            from sklearn.neural_network import MLPClassifier
            MLP_clf = MLPClassifier(solver='adam',
                                    activation='relu',
                                    max_iter=5000, 
                                    early_stopping=True, 
                                    random_state=2023)
            param_grid = {
                'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                'hidden_layer_sizes': [(32,),(64,),(128,),(32,32),(64,64),(128,128)],
            }
            search = GridSearchCV(estimator=MLP_clf, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True)
            search.fit(X_train, y_train)
            MLP_clf = search.best_estimator_
            ret = get_algorithm_result(
                X_train,y_train,
                X_test,y_test,
                X_external,y_external,
                target_names,
                classifier=MLP_clf, 
                clf_name='MLP',
                dir_result=dir_result)
            ret['search'] = search
            MLP_clf = ret['classifier']
            result_dict['MLP'] = ret
    return result_dict

