import sys
import numpy as np
import pandas as pd

import xgboost
from xgboost.sklearn import XGBClassifier

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import cPickle as cPickle
from pybedtools import BedTool

import copy
import scipy.stats as st
import datetime
import argparse

from matplotlib.lines import Line2D
matplotlib.rc('xtick', labelsize=10) 


def fill_nan(s):
    '''
    Replaces '.' with np.nan
    
    parameters:
        @s: pandas series
    returns:
        @s_median: pandas series with missing values filled
    '''
    missing_indices = s.index[s=='.']
    s.iloc[missing_indices] = np.repeat(np.nan,len(missing_indices))


    return s



def clean_df(df):
	type_groups = df.columns.to_series().groupby(df.dtypes).groups
	type_dict = {k.name: v for k, v in type_groups.items()}

	for feature in type_dict['object']:
		if feature in ['chr','start','end']:
			continue
		df[feature] = fill_nan(df[feature])
	
	return df


def get_data(clean=False):

	#path to input directory and files for training and testing.
	indir = 'centromere_data_030319/' 
        x_test_p,x_train_p,y_test_p,y_train_p = 'x_test_allMatched030319_v11.txt','x_train_allMatched030319_v11.txt','y_test_allMatched030319_v11.txt', 'y_train_allMatched030319_v11.txt'


	x_train_path = indir+x_train_p
	x_test_path = indir+x_test_p
	y_train_path = indir+y_train_p
	y_test_path = indir+y_test_p


	if clean is False:

		x_train_df = pd.read_table(x_train_path,sep='\t',index_col=0,header=0,dtype=None,na_values='.')
		x_train = pd.read_table(x_train_path,sep='\t',index_col=0,header=0,dtype=None,na_values='.').as_matrix()
		x_test = pd.read_table(x_test_path,sep='\t',index_col=0,header=0,dtype=None,na_values='.').as_matrix()
		y_train = pd.read_table(y_train_path,sep='\t',index_col=0,header=0,dtype=None,na_values='.')
                y_test = pd.read_table(y_test_path,sep='\t',index_col=0,header=0,dtype=None,na_values='.')


		y_train = y_train.astype({'label':int})
		y_test = y_test.astype({'label':int})

		y_train = y_train.as_matrix()
		y_test = y_test.as_matrix()

	if clean is True:

                x_train_df = clean_df(pd.read_table(x_train_path,sep='\t',index_col=0,header=0,dtype=None))
                x_train = clean_df(pd.read_table(x_train_path,sep='\t',index_col=0,header=0,dtype=None)).as_matrix()
                x_test = pd.read_table(x_test_path,sep='\t',index_col=0,header=0,dtype=None).as_matrix()
                y_train = clean_df(pd.read_table(y_train_path,sep='\t',index_col=0,header=0,dtype=None)).as_matrix()
                y_test = pd.read_table(y_test_path,sep='\t',index_col=0,header=0,dtype=None).as_matrix()

	return x_train_df,x_train,x_test,y_train,y_test

def hypertune(x_train_df,x_train,x_test,y_train,y_test,n_iter=100,cv=5):

	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 32, stop = 96, num=4)]
	
	# Maximum number of levels in tree
	max_depth = st.randint(10,41)
	
	learning_rate = st.uniform(0.0005,0.5)
	gamma = st.uniform(0,10)

	one_to_left = st.beta(10, 1)  
	from_zero_positive = st.expon(0, 50)

	scale_pos_weight = [1,2,3,4,5,6,7,8,9,10]

	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
		'learning_rate':learning_rate,
		"colsample_bytree": one_to_left,
		"subsample": one_to_left,
		'reg_alpha': from_zero_positive,
		"min_child_weight": from_zero_positive,
		'gamma':gamma,
		'scale_pos_weight':scale_pos_weight}

	xgb_tune = xgboost.XGBClassifier(missing=np.nan)
	xgb_hyper = RandomizedSearchCV(xgb_tune,random_grid, 
                               n_iter = n_iter, cv = cv, random_state=42, n_jobs = -1,scoring='roc_auc')


	# Fit the random search model
	xgb_hyper.fit(x_train, y_train.ravel())
	return  xgb_hyper.best_estimator_

def plot_roc_curve(y_pred,y_test,title,text,outname):
 	'''
	Plots a ROC curve for given set of predictions
    
	parameters:
	@predict_results: predict generator object
	return:
	None
	'''
    
	y = np.asarray(y_test).reshape((len(y_test),1))
	y_hat = np.asarray(y_pred).reshape((len(y_pred),1))
    
	fpr, tpr, _ = roc_curve(y,y_hat)

	curr_time = datetime.datetime.now()

	AUROC = auc(fpr,tpr)
	plt.plot(fpr,tpr,label=text + ' (AUC = %.3f)' % AUROC)
	plt.legend(loc='lower right')
	plt.title(title)
	plt.xlabel('1 - Specificity')
	plt.ylabel('Sensitivity')
	plt.savefig(image_dir+'ROC_CURVE_%s.png'%(outname))
    	plt.close()
	return

def plot_pr_curve(y_pred,y_test,title,text,outname):
	'''
	Plots a PR Curve for a given set of predictions
     	
	parameters:
		@predict_results: predict generator object	
	return:
		None
	'''    

	y_hat = y_pred    
	y = np.asarray(y_test).reshape((len(y_test),1))
	y_hat = np.asarray(y_hat).reshape((len(y_hat),1))

	precision, recall, thresholds = precision_recall_curve(y, y_hat)
	AUPR = average_precision_score(y,y_hat)
    
	plt.plot(recall,precision,label=text + ' (AP = %.3f)' % AUPR,color='r')
	plt.legend(loc='lower left')
	plt.title(title)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.savefig(image_dir+'PR_CURVE_%s.png'%(outname))    
	plt.close()
	return

def gather_scores(x_test,y_test,rf_results,features):
	'''
	Gathers all genomewide scoring metric scores for each variant in test set along with
	the deep learning model scores on the test set.
    
	parameters:
		@x_test: features in test set
		@y_test: labels in test set
		@prediction_results: prediction results object
		@rf_results: random forest predictions
		@features: list of feature names
	returns:
		None
	'''
    
	x_df = pd.DataFrame(x_test,columns=features)	
	x_df = x_df[['linsight','cadd','eigen','funseq','ncRVIS','orion','fathmm','remm']]
	x_df['xgb'] = rf_results
	x_df['label'] = y_test
	return x_df


def compare_PR(scores_df,scores,title,outname):
	'''
	Plots a PR curve for all scores on same plot.
    
	parameters:
		@scores_df: pandas dataframe that contains scores and label for each variant
	returns:
	None
	'''
	colors = ['r','g','b','k','c','m','y','orange','grey','pink','crimson']
    
	sc = zip(scores,colors[0:len(scores)])
    
	for tup in sc:
		score,color = tup
		scores_df.sort_values(by=score,ascending=False,inplace=True)
		
		if score == 'cdts' or score=='ncRVIS': #Lower CDTS score --> more likely pathogenic
			scores_df.sort_values(by=score,ascending=True,inplace=True) 
		
		
		totalPath =  len(scores_df[scores_df['label'] == 1])
 		totalControl = len(scores_df[scores_df['label'] == 0])
		scores_df.index = range(len(scores_df.index))
		x,y = [],[]
		
		for i,row in scores_df.iterrows():
			t_s = scores_df[scores_df.index<=i]
			num_p = len(t_s[t_s['label']==1])
			num_c = len(t_s[t_s['label']==0])
       			recall = float(num_p)/(totalPath)
			precision = float(num_p)/(num_p+num_c)
			x.append(recall)
			y.append(precision)
        	score_auc = auc(x,y)
		plt.plot(x,y,label=score+' (AUC: %.3f)'%score_auc,color=color)
        
	plt.title(title)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(bbox_to_anchor=(1,0.9))
	plt.tight_layout()
	plt.savefig(image_dir+'PR_CURVE_Comparison_%s.png'%(outname))
	plt.close()
	return



def compare_ROC(scores_df,scores,title,outname):
	'''
	Plots a ROC curve for all scores on same plot.
    
	parameters:
		@scores_df: pandas dataframe that contains scores and label for each variant
	returns:
	None
	'''
	colors = ['r','g','b','k','c','m','y','orange','grey','pink','crimson']
    
    	sc = zip(scores,colors[0:len(scores)])
    
	for tup in sc:	
		score,color = tup
		scores_df.sort_values(by=score,ascending=False,inplace=True)
		if score == 'cdts' or score == 'ncRVIS': #Lower CDTS score --> more likely pathogenic
			scores_df.sort_values(by=score,ascending=True,inplace=True) 
		totalPath =  len(scores_df[scores_df['label'] == 1])
		totalControl = len(scores_df[scores_df['label'] == 0])
		scores_df.index = range(len(scores_df.index))
		x,y = [],[]
		
		for i,row in scores_df.iterrows():
			t_s = scores_df[scores_df.index<=i]
			num_p = len(t_s[t_s['label']==1])
			num_c = len(t_s[t_s['label']==0])
        
			x.append(float(num_c)/totalControl)
			y.append(float(num_p)/totalPath)
        
		score_auc = auc(x,y)
		plt.plot(x,y,label=score+' (AUC: %.3f)'%score_auc,color=color)
   
	plt.title(title)
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend(bbox_to_anchor=(1,0.9))
    
	plt.tight_layout()
	plt.savefig(image_dir+'ROC_CURVE_Comparison_%s.png'%(outname))
	plt.close()
    
	return


def plotFI(features,importances,indices,bar_colors,outname):
	'''
	Plots a bar chart for the Random Forest Feature Importances
    
	parameters:
		@features: list of feature names
		@importances: list of importances
		@indices: ordering of importances
		@std: error bars
	returns:
	None
	'''	
	
	legend_elements = [Line2D([0], [0], color='g', lw=2, label='essentiality'),Line2D([0], [0], color='b', label='scores',lw=2),Line2D([0],[0],color='orange',label='regulatory',lw=2),Line2D([0],[0],color='red',label='structure')]

	orderedImportances = [features[ind] for ind in indices]
	orderedScores = [importances[ind] for ind in indices]

	width = 0.35
	xInd = np.arange(len(orderedScores))[0:len(importances)]	
	fig,ax = plt.subplots(figsize=(10,10))	
	rects = ax.bar(xInd,orderedScores[0:len(importances)],color=bar_colors)
	ax.set_ylabel('Feature Importance')
	ax.set_xticks(xInd + width / 2)
	ax.set_xticklabels(orderedImportances, rotation=90)
	ax.set_title('XGBoost Feature Importances')	
	ax.legend(handles=legend_elements, loc='upper right')
	fig.tight_layout()
	plt.savefig(image_dir+'Feature_Importances_%s.png'%(outname))
	plt.close()
	
	return

def get_subsets():
	'''
	Group features into broad categories
	'''

	scoring_features = ['linsight','cadd','eigen','funseq','fathmm','remm','ncRVIS','orion']

	three_d_features = ['combinedAnchors','combinedDomains','combinedLoops','fire','pchic-pLI','pchic-binary','enhancerTSS','enhancerTSS-binary','MNaseGSM920558','MNaseGSM920557','TADS']

	regulatory_features = ['logMedExpMedGene','logStdExp','logMedExpMedian','inhibitorPeaks','noInhibitorPeaks','superEnhancernaive','superEnhancerprimed','vistaCelltypes','vistaMeanActivity','vistaActivity','STARRnaive','STARRprimed','fantomPlncRNA','fantomRlncRNA','fantomSlncRNA']

	gene_essentiality_features = ['cdts','geneEssentiality','haploI','omim','dosage','AD','AR']


	subset_data = [scoring_features,regulatory_features,gene_essentiality_features,three_d_features]
	subset_names = ['scores','regulatory','essentiality','structure']

	return subset_data,subset_names


def plot_feature_importances(xgb_model,df,outname):
	'''
	Plot feature importances for a given model
	'''

	subset_data,subset_names = get_subsets()

	rankings = []    
	importances = xgb_model.feature_importances_

	indices = np.argsort(importances)[::-1]
	print ("Feature Ranking:")
	for f in range(len(list(importances))):
    		print ('(%d) %s : %f' % (f+1,list(df)[indices[f]],importances[indices[f]]))
    		rankings.append(list(df)[indices[f]])
	

	color_dict = {}
	colors = ['b','orange','g','r','purple','sienna','k']
	bar_colors = []
	for i,sd in enumerate(subset_data):
		bar_c = colors[i]
		for s in sd:
			color_dict[s] = bar_c

	color_order = []
	for feat in rankings:
		color_order.append(color_dict[feat])
	
	plotFI(list(df),importances,indices,color_order,outname)
	
	return


def autolabel(rects,ax):
	'''
	Labels a bar chart
	'''
	
	for rect in rects:
		h = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2.,1.01*h,'%.2f'%float(h),ha='center',va='bottom')

	return


def keep_subset(df,subset):
	'''
	Keeps only a subset of features for model training
	'''
	
	x_subset = df[subset]

	return x_subset




def subset_analysis_paper(x_train,y_train,x_test,y_test,df,xgb_model_path,outname):
	'''
	Feature subset analysis
	Compares performance of a model trained on only the scoring metrics, on all features except the scoring metrics, and 
	a model trained using all features.
	'''

	subset_data,subset_names = get_subsets()
	with open(xgb_model_path,'rb') as fp:
		xgb_full_model = cPickle.load(fp)
	
	params = xgb_full_model.get_params()
	subset_performance_roc = {}
	subset_performance_pr = {}
	

	name_to_idx_dict = dict(zip(list(df),np.arange(0,len(list(df)))))

	total_feats_to_keep = []
	subset_string_arr = []
	total_feat_string = ''
	count = 1

	#scoring metrics only 
	x_subset_scores = keep_subset(df,subset_data[0]).values
	x_test_scores = copy.deepcopy(x_test)
	keep_cols = [name_to_idx_dict[n] for n in subset_data[0]]
	x_test_scores = x_test_scores[:,keep_cols]
	xgb_best_scores = hypertune(df,x_subset_scores,x_test_scores,y_train,y_test)
	xgb_scores_preds = xgb_best_scores.predict_proba(x_test_scores)[:,1]

	#everything but scoring
	keep_everything_but_scores = subset_data[1] + subset_data[2] + subset_data[3]
	x_subset_other = keep_subset(df,keep_everything_but_scores).values
	x_test_other = copy.deepcopy(x_test)
	keep_cols = [name_to_idx_dict[n] for n in keep_everything_but_scores]
        x_test_other = x_test_other[:,keep_cols]
        xgb_best_other = hypertune(df,x_subset_other,x_test_other,y_train,y_test)
	xgb_other_preds = xgb_best_other.predict_proba(x_test_other)[:,1]


	xgb_full_preds = xgb_full_model.predict_proba(x_test)[:,1]

	#compare performance
	fpr_scores,tpr_scores,_ = roc_curve(y_test,xgb_scores_preds)
	fpr_other,tpr_other,_ = roc_curve(y_test,xgb_other_preds)
	fpr_full,tpr_full,_ = roc_curve(y_test,xgb_full_preds)

	AUROC_scores = auc(fpr_scores,tpr_scores)
	AUROC_other = auc(fpr_other,tpr_other)
	AUROC_full = auc(fpr_full,tpr_full)


	prec_scores,rec_scores,_ = precision_recall_curve(y_test,xgb_scores_preds)
	prec_other,rec_other,_ = precision_recall_curve(y_test,xgb_other_preds)
	prec_full,rec_full,_ = precision_recall_curve(y_test,xgb_full_preds)	


	AUPR_scores = auc(rec_scores,prec_scores)
	AUPR_other = auc(rec_other,prec_other)
	AUPR_full = auc(rec_full,prec_full)
	
	subset_performance_roc['scoring metrics'] = AUROC_scores
	subset_performance_roc['essentiality + regulatory + structure'] = AUROC_other
	subset_performance_roc['full model'] = AUROC_full


	subset_performance_pr['scoring metrics'] = (prec_scores,rec_scores,AUPR_scores)
	subset_performance_pr['essentiality + regulatory + structure'] = (prec_other,rec_other,AUPR_other)
	subset_performance_pr['full model'] = (prec_full,rec_full,AUPR_full)
	

	plt.figure(figsize=(8,6))
	plt.plot(fpr_scores,tpr_scores,label='scoring metrics' + ' (AUROC: %.3f)' % (AUROC_scores))
	plt.plot(fpr_other,tpr_other,label='essentiality + regulatory + structure' + ' (AUROC: %.3f)' % (AUROC_other))
	plt.plot(fpr_full,tpr_full,label='full model' + ' (AUROC: %.3f)' % (AUROC_full))        

	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend(loc='lower left',fontsize=6,bbox_to_anchor=(1,0.5))
	plt.title('XGB ROC Feature Set Comparison')
	plt.tight_layout()
	plt.show()
	plt.savefig(image_dir+'ROC_subsets_%s_paper.png'%outname)
	plt.close()


	plt.figure(figsize=(8,6))
	plt.plot(rec_scores,prec_scores,label='scoring metrics'+' ( AUPR %.2f)' % (AUPR_scores))
	plt.plot(rec_other,prec_other,label='essentiality + regulatory + structure'+' ( AUPR %.2f)' % (AUPR_other))
	plt.plot(rec_full,prec_full,label='full model'+' ( AUPR %.2f)' % (AUPR_full))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left',fontsize=6,bbox_to_anchor=(1,0.5))
        plt.title('XGB PR Feature Sets')
	plt.tight_layout()
        plt.show()
        plt.savefig(image_dir+'PR_subsets_%s_paper.png'%outname)
        plt.close()


	return



def test_model(xgb_mod,x_train,y_train,x_test,y_test,mod,df,args):
	'''
	Evaluate model performance
		ROC and PR curves
	Subset of features analysis
	Feature importance analysis
	'''
	outstring = mod.split('.')[1]

	preds = xgb_mod.predict_proba(x_test)[:,1]
	

	if args.ROC_PR_FLAG == 'True':
		plot_roc_curve(preds,y_test.ravel(),'XGBoost ROC Curve','XGBoost',outstring)
		plot_pr_curve(preds,y_test.ravel(),'XGBoost PR Curve','XGBoost',outstring)	

		scores_df = gather_scores(x_test,y_test,preds,list(df))
	
		compare_ROC(scores_df,['linsight','orion','cadd','eigen','funseq','ncRVIS','fathmm','remm','xgb'],'ROC Curve Comparison',outstring)
		compare_PR(scores_df,['linsight','orion','cadd','eigen','funseq','ncRVIS','fathmm','remm','xgb'],'PR Curve Comparison',outstring)

	if args.feature_importance == 'True':
		plot_feature_importances(xgb_mod,df,outstring)

	if args.subset_analysis_paper == 'True':
		subset_analysis_paper(x_train,y_train,x_test,y_test,df,mod,outstring)


	return


def main(args):
	'''
	Tune, evalute, and train a XGBoost model on a provided dataset.
	'''

	if args.clean == 'True': #if data preprocessing needs to be completed
		x_train_df,x_train,x_test,y_train,y_test = get_data(clean=True)
	else:
		x_train_df,x_train,x_test,y_train,y_test = get_data()

	if args.hyper_tune == 'True': #train an XGBoost model
		np.random.seed(1)
		xgb_best = hypertune(x_train_df,x_train,x_test,y_train,y_test)
	
		curr_time = datetime.datetime.now()
		model_file_name = 'xgb_best_%s.cPickle' %(curr_time)
		with open(args.model_dir+model_file_name,'wb') as fp:
			cPickle.dump(xgb_best,fp)
	
		mod_string = str(curr_time)	

	else:
		with open(args.model,'rb') as fp:
			xgb_best = cPickle.load(fp)
		mod_string = args.model

	#Evaluate the model on validation sets.  Plot feature importance, subset analysis
	test_model(xgb_best,x_train,y_train,x_test,y_test,mod_string,x_train_df,args)

	return





if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--hyper_tune',type=str,default='True')	
	parser.add_argument('--model',type=str,default='')
	parser.add_argument('--ROC_PR_FLAG',type=str,default='True')
	parser.add_argument('--feature_importance',type=str,default='True')
	parser.add_argument('--subset_analysis_paper',type=str,default='True')
	parser.add_argument('--image_dir',type=str,default='images_021819/')
	parser.add_argument('--model_dir',type=str,default='models_021819/')
	parser.add_argument('--clean',type=str,default='False')

	args = parser.parse_args()

	image_dir = args.image_dir
	main(args)



