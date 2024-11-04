import matplotlib.pyplot as plt
import os

def loss_plot(epoch,loss,plot_save_path):
    num = epoch
    x = [i for i in range(num)]
    # plot_save_path = r'../result/plot_dim_sim4/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+'_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

def metrics_plot(plot_save_path, epoch,name,*args):
    num = epoch
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    # plot_save_path = r'../result/plot_dim_sim4/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)