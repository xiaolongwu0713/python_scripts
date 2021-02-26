
import matplotlib.pyplot as plt
def plot_on_test(ax ,targets ,preds ,num):
    plt.cla()
    flat_t = [item for sublist in targets for item in sublist]
    flat_p = [item for sublist in preds for item in sublist]
    ax.plot(flat_t, color="orange")
    ax.plot(flat_p, 'g-', lw=3, label=num)
    filename ='test ' +str(num ) +'.png'
    plt.savefig(filename)

fig, ax = plt.subplots(figsize=(6,3))
for i in range(5):
  a=[[1,2,3,4,5],[6,7,8,9,0],[3,4,5,6,7]]
  b=2*a
  plot_on_test(ax,a,b,i)

