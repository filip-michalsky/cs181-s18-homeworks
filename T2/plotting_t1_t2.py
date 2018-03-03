#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:17:27 2018

@author: filipmichalsky
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_post_betas(n1,n0,x,alpha=4,beta=2,save_fig = False,cnt=0,ax=None,points=[]):
    
    #this will be p(theta|x) -> posterior
    beta_dist = lambda x: (x**(n1+alpha-1))*(1-x)**(n0+beta-1)
    if cnt !=0:
        ax.plot(x,beta_dist(x),label=("Posterior Beta Distribution ({},{})".format(n1+alpha,n0+beta)));
    else:
        ax.plot(x,beta_dist(x),label=("Initial Beta Distribution ({},{})".format(n1+alpha,n0+beta)));
    if (n1+n0)!=0:
        theta_MLE = n1/(n1+n0)
        #if theta_MLE == 0:
            #ax.scatter(theta_MLE,beta_dist(theta_MLE),label="theta_MLE",color='red');
    
        #else:
            #ax.vlines(theta_MLE,ymin=0,ymax=beta_dist(theta_MLE),colors='red',label="theta_MLE",linewidth=1.5);
    
    if n0!=0:
        theta_MAP = (n1+alpha-1)/(n1+n0+alpha+beta-2)
        #ax.vlines(theta_MAP,ymin=0,ymax=beta_dist(theta_MAP),color='green',label="theta_MAP");
    
    
    theta_PP = (alpha+n1)/(alpha+beta+n1+n0)
    #ax.vlines(theta_PP,ymin=0,ymax=beta_dist(theta_PP),color='magenta',label="theta_PP")
    ax.set_ylim(0,beta_dist(theta_PP)+beta_dist(theta_PP)*0.2)
    ax.set_xlim(-0.1,1.2);
    plt.tight_layout();
    
    ax.legend();

    ax.set_title("# points: {}, Data points {}".format(len(points),points))
    
    if save_fig:
        plt.savefig("Thetas {}".format(cnt))
        
def plot_thetas(n1,n0,x,alpha=4,beta=2,save_fig = False,cnt=0,ax=None,points=[]):
    
    #this will be p(theta|x) -> posterior
    beta_dist = lambda x: (x**(n1+alpha-1))*(1-x)**(n0+beta-1)
    if cnt !=0:
        ax.plot(x,beta_dist(x),label=("Posterior Beta Distribution ({},{})".format(n1+alpha,n0+beta)));
    else:
        ax.plot(x,beta_dist(x),label=("Initial Beta Distribution ({},{})".format(n1+alpha,n0+beta)));
    
    if (n1+n0)!=0:
        theta_MLE = n1/(n1+n0)
        if theta_MLE == 0:
            ax.scatter(theta_MLE,beta_dist(theta_MLE),label="theta_MLE",color='red');
    
        else:
            ax.vlines(theta_MLE,ymin=0,ymax=beta_dist(theta_MLE),colors='red',label="theta_MLE",linewidth=1.5);
    
    if n0!=0:
        theta_MAP = (n1+alpha-1)/(n1+n0+alpha+beta-2)
        ax.vlines(theta_MAP,ymin=0,ymax=beta_dist(theta_MAP),color='green',label="theta_MAP");
    
    
    theta_PP = (alpha+n1)/(alpha+beta+n1+n0)
    ax.vlines(theta_PP,ymin=0,ymax=beta_dist(theta_PP),color='magenta',label="theta_PP")
    ax.set_ylim(0,beta_dist(theta_PP)+beta_dist(theta_PP)*0.2)
    ax.set_xlim(-0.1,1.2);
    plt.tight_layout();
    
    ax.legend();

    ax.set_title("# points: {}, Data points {}".format(len(points),points))
    
    if save_fig:
        plt.savefig("Thetas {}".format(cnt))
        
x = np.linspace(0,1,50)

fig, axes = plt.subplots(9, 2,figsize=(15,35));

alpha = 4
beta = 2
n1 = 0
n0 = 0
cnt = 0
points_display =[]

points = [None,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0]

row,col = (0,0)

for point in points:

    if point == 1:
        n1+=1
    elif point == 0:
        n0+=1
    
    points_display.append(point)
    plot_thetas(n1=n1,n0=n0,x=x,alpha=4,beta=2,cnt=cnt,save_fig=False,ax=axes[row,col],points=points_display);
    
    if col % 2 == 0:
        col = 1
    else:
        col=0
        row+=1
        
    cnt+=1

plt.savefig("Problem_1")

x = np.linspace(0,1,50)

fig, axes = plt.subplots(3, 2,figsize=(15,15));

alpha = 4
beta = 2
n1 = 0
n0 = 0
cnt = 0
points_display =[]

points = [None,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0]
checks = [0,4,8,12,16]
row,col = (0,0)

for point in points:

    if point == 1:
        n1+=1
    elif point == 0:
        n0+=1
    if point != None:
        points_display.append(point)
    #print(len(points_display))
    if len(points_display) in checks:
        plot_post_betas(n1=n1,n0=n0,x=x,alpha=4,beta=2,cnt=cnt,save_fig=False,ax=axes[row,col],points=points_display);
    
        if col % 2 == 0:
            col = 1
        else:
            col=0
            row+=1
    
    cnt+=1

plt.savefig("Problem_1_5")