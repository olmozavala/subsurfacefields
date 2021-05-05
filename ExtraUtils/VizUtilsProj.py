import cmocean
import matplotlib.pyplot as plt
import numpy as np

def chooseCMAP(fields):
    cmaps_fields = []
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            cmaps_fields.append(cmocean.cm.deep_r)
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            cmaps_fields.append(cmocean.cm.thermal)
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            cmaps_fields.append(cmocean.cm.haline)
        elif c_field == "u-vel.":
            cmaps_fields.append(cmocean.cm.delta)
        elif c_field == "v-vel.":
            cmaps_fields.append(cmocean.cm.delta)
        else:
            cmaps_fields.append(cmocean.cm.thermal)
    return cmaps_fields


def getMinMaxPlot(fields):
    minmax = []
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            minmax.append([-4, 4])
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            minmax.append([-2, 2])
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            minmax.append([-.2, .2])
        elif c_field == "u-vel":
            minmax.append([-1, 1])
        elif c_field == "v-vel":
            minmax.append([-1, 1])
        else:
            minmax.append(cmocean.cm.thermal)
    return minmax


def _draw_profile(t, s, depth, ax1):
    t_color = 'r'
    s_color = 'y'

    if len(t.shape) > 1:
        max_depth_i = 0
        for i in range(t.shape[0]):
            c_max_depth = np.min(np.where(np.isnan(t[i,:]))) - 1
            if c_max_depth > max_depth_i:
                max_depth_i = c_max_depth

        t = t[:, 0:max_depth_i]
        s = s[:, 0:max_depth_i]
        depth = depth[0:max_depth_i]

        ax1.scatter(t.flatten(), np.tile(depth, t.shape[0]),  s=.6, c=t_color, alpha=0.7)
        ax1.plot(np.mean(t,axis=0), depth, c='k')
    else:
        ax1.plot(t, depth, t_color)

    ax1.set_xlabel('Temp', color=t_color)
    ax1.tick_params(axis='x', labelcolor=t_color)
    # ax1.set_xlim([0,40])
    ax1.invert_yaxis()

    ax2 = ax1.twiny()
    if len(t.shape) > 1:
        ax2.scatter(s.flatten(), np.tile(depth, t.shape[0]),  s=.6, c=s_color, alpha=0.4)
        ax2.plot(np.mean(s,axis=0), depth, c='b')
    else:
        ax2.plot(s, depth, s_color)
    ax2.set_xlabel('Salinity', color=s_color)
    # ax2.set_xlim([30,40])
    ax2.tick_params(axis='x', labelcolor=s_color)


def draw_profile(t, s, depth, title, output_file = ""):
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(111)
    plt.title(title)

    _draw_profile(t, s, depth, ax1)

    if output_file != "":
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def draw_profiles_comparison(t, s, nn_t, nn_s, depth, title, output_file = ""):
    fig = plt.figure(figsize=(40,10))
    plt.title(title)
    axt = fig.add_subplot(121)
    _draw_profile(t, s, depth, axt)
    axnn = fig.add_subplot(122)
    _draw_profile(nn_t, nn_s, depth,  axnn)

    if output_file != "":
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
