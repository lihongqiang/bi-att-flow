#encoding:utf-8
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import sys
def str2vec(l,sep=" "):
	return [float(f) for f in l.split(sep)]
def gen_cos_result(q_file_path,p_file_path,orig_file_path,target_file_path):
	s2v = {}
	with open(q_file_path,encoding='utf-8') as qfile,open(p_file_path,encoding='utf-8') as pfile:
		for l in qfile:
			fields = l.replace("\n","").split("\t")
			s2v[fields[0]]=str2vec(fields[1],",")
		for l in pfile:
			fields = l.replace("\n","").split("\t")
			s2v[fields[0]]=str2vec(fields[1],",")
	with open(orig_file_path,encoding='utf-8') as origfile,open(target_file_path,"w",encoding='utf-8') as targetfile:
		for l in origfile:
			fields = l.replace("\n","").split("\t")
			q = fields[0]
			p = fields[1]
			t = fields[2]
			s = cosine_similarity([s2v[q]],[s2v[p]])[0,0]
			targetfile.write("{}\t{}\t{}\t{}\n".format(q,p,t,s))
def load_dense_file(file_path):
	lines = None
	with open(file_path) as infile:
		lines = infile.readlines()
	m,n = lines[0].strip().split("\t")
	m=int(m)
	n=int(n)
	matrix = np.zeros(shape=(m,n))
	for i in range(1,1+m):
		x=np.array([float(f) for f in lines[i].strip().split("\t")])
		matrix[i-1]=x
	b=np.array([float(f) for f in lines[m+2].strip().split("\t")])
	return matrix,b
def mlp_score(q,p,w,b):
	qq = np.array(q)
	pp = np.array(p)
	dot = qq*pp
	matmul = np.matmul(dot,w)
	add_bias = matmul+b
	add_bias = np.exp(add_bias)
	add_bias = add_bias/np.sum(add_bias)
	return add_bias
def compute_vector_mlp(q_path,p_path,t_path,target_path,w,b):
	qs = {}
	ps = {}
	tags = {}
	with open(q_path,encoding="utf-8") as qfile,open(p_path,encoding="utf-8") as pfile,open(t_path,encoding="utf-8") as tfile,open(target_path,"w",encoding="utf-8") as targetfile:
		for q in qfile:
			q=q.replace("\n","")
			fields = q.split("\t")
			qvec = str2vec(fields[1],",")
			qs[fields[0]]=qvec
		for p in pfile:
			p=p.replace("\n","")
			fields = p.split("\t")
			pvec = str2vec(fields[1],",")
			ps[fields[0]]=pvec
		for t in tfile:
			t=t.replace("\n","")
			fields = t.split("\t")
			tag = fields[2]
			#try:
			s = mlp_score(qs[fields[0].strip()],ps[fields[1].strip()],w,b)[1]
			targetfile.write("{}\t{}\n".format(tag,s))
			#except:pass
if __name__ == '__main__':
	qfile = sys.argv[1]
	pfile = sys.argv[2]
	origfile = sys.argv[3]
	targetfile = sys.argv[4]
	#gen_cos_result(qfile,pfile,origfile,targetfile)
	w,b = load_dense_file("dense_w_b")
	compute_vector_mlp(qfile,pfile,origfile,targetfile,w,b)
