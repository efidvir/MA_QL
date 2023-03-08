import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import pygame
import cv2
import glob
import os
plt.rcParams["figure.dpi"] = 100
from itertools import cycle
viridis = cm.get_cmap('viridis', 8)

class render():
    def __init__(self):
        pass

    #def policy(self,policy):
    #    plt. plot(range(len(policy),eee0, linewidth=1)

    def last_1k_slots(self,data, number_of_agents):
        data = np.reshape(data, (100, 100))
        #print(data)

        # create discrete colormap
        viridis = cm.get_cmap('gist_rainbow', 256-2*int(256 / (number_of_agents+2)))
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[2*int(256 / (number_of_agents+2)) +1:, :] = viridis(np.linspace(0, 1, len(newcolors[2*int(256 / (number_of_agents+2)) +1:, :])))
        newcolors[:int(256/(number_of_agents+2)), :] = np.array([1, 1, 1, 1])#wasted = white
        newcolors[int(256/(number_of_agents+2)):2*int(256 / (number_of_agents+2))+1 , :] = np.array([0, 0, 0, 1])#collision = black
        #newcolors[255:256, :] = np.array([0.75, 0.75, 0.75, 1])  # last agent = grey
        cmap = ListedColormap(newcolors)
        bounds = np.linspace( -0.1, number_of_agents+2,number_of_agents+3)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.01)
        ax.set_xticks(np.arange(1, 100, 1));
        ax.set_yticks(np.arange(1, 10, 1));
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.grid(None)
        labels = ["Wasted", "Collision"]
        for i in range(number_of_agents):
            labels.append("Agent {}".format(i+1))
        patches =[mpatches.Patch(color= cmap(norm(i)),label=labels[i]) for i in range(number_of_agents+2)]
        ax.legend(title = "Time slot usage", handles=patches, loc=4, prop={'size': 6},bbox_to_anchor=(1,1.2),ncol=3, fancybox=True, shadow=True)
        #ax.title('Last 1000 time steps')
        plt.savefig('1000.pdf', dpi=1000)
        plt.show()

    def plot_Q_values(self, tables_array, number_of_iterations):
        for i in range(np.shape(tables_array)[1]):
            plt.plot(range(number_of_iterations), tables_array[i,])
        plt.show()

    def render_Q_diffs(self, Q1, Q2, agent_num,iteration,state, action, reward, next_state):
        path = 'C:/Users/dvire/PycharmProjects/MA_QL/images/'
        screen = pygame.display.set_mode((Q1.shape[0] * 100, Q1.shape[1] * 100))
        diff = (Q1 - Q2)
        diff_pos = diff - np.min(diff)  # shift to posetive
        max = np.max(diff_pos)
        if max == 0:
            max = 1
        color = diff_pos / max * 255
        current_energy, slient_time = state
        next_energy, next_slient_time = next_state
        #print('Difference Q1 - Q2 (wait - transmit)')
        for i in range(Q1.shape[0]):
            for j in range(Q1.shape[1]):
                pygame.draw.rect(screen, 255 - int(color[i, j]), pygame.Rect(i * 100, j * 100, 100, 100))
                if i == current_energy and j == slient_time:
                    pygame.draw.rect(screen, (255,0,0), pygame.Rect(i * 100, j * 100, 100, 100))
                    #arrow start point
                    start_point = ((i+1) * 100 -50, (j+1) * 100 -50)
                if i == next_energy and j == next_slient_time:

                    pygame.draw.rect(screen, (255, 128, 0), pygame.Rect(i * 100, j * 100, 100, 100))
                    #arrow end coordinate
                    end_point = ((i+1) * 100 -50, (j+1) * 100 -50)
        pygame.display.flip()

        # convert image so it can be displayed in OpenCV
        view = pygame.surfarray.array3d(screen)

        #  convert from (width, height, channel) to (height, width, channel)
        view = view.transpose([1, 0, 2])

        #  convert from rgb to bgr
        img = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        for i in range(Q1.shape[0]):
            for j in range(Q2.shape[1]):
                img = cv2.putText(img, "%.2f " % diff[i, j], (i * 100 + 20, j * 100 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 1, cv2.LINE_AA)
                img = cv2.putText(img, "E = %d" % i, (i * 100, j * 100 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "T = %d" % j, (i * 100, j * 100 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "Q1 = %.3f" % Q1[i, j], (i * 100, j * 100 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "Q2 = %.3f" % Q2[i, j], (i * 100, j * 100 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 255), 1, cv2.LINE_AA)

        img = cv2.arrowedLine(img, start_point, end_point, (128, 255, 128), 3)
        img = cv2.putText(img, "a=%d" % action, (int((start_point[0] +end_point[0])/2),int((start_point[1] +end_point[1])/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 0, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, "r=%d" % reward, (int((start_point[0] +end_point[0])/2),int((start_point[1] +end_point[1])/2)+20) , cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 128, 255), 2, cv2.LINE_AA)

                # img_bgr = cv2.putText(img_bgr, "%.3f" % value[i,j], (i*100,j*100+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # else:
        #  img = cv2.putText(img_bgr, "Wait", (i*100,j*100+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.imshow('Q DIffs agent{d}'.format(d=agent_num),img)
        img = img[0:Q1.shape[0] * 100, 0:Q1.shape[1] * 100]
        path = os.path.join(path, 'agent{d}'.format(d=agent_num))
        cv2.imwrite(os.path.join(path , 'Q_DIffs_agent{d}_{e}.jpg'.format(d=agent_num, e = iteration)), img)
        #out = cv2.VideoWriter('Q_DIffs_agent{d}.avi'.format(d=agent_num), cv2.VideoWriter_fourcc(*'DIVX'),10,(Q1.shape[0] * 100, Q1.shape[1] * 100))
        #out.write(img)
        #out.release()

    def render_Q(self, Q1, Q2, agent_num,iteration,state):
        #path = 'C:/Users/dvire/PycharmProjects/MA_QL/images/'
        screen = pygame.display.set_mode((Q1.shape[0] * 100, Q1.shape[1] * 100))
        diff = (Q1 - Q2)
        diff_pos = diff - np.min(diff)  # shift to posetive
        max = np.max(diff_pos)
        if max == 0:
            max = 1
        color = diff_pos / max * 255
        current_energy, slient_time = state
        #print('Difference Q1 - Q2 (wait - transmit)')
        for i in range(Q1.shape[0]):
            for j in range(Q1.shape[1]):
                pygame.draw.rect(screen, 255 - int(color[i, j]), pygame.Rect(i * 100, j * 100, 100, 100))
                if i == current_energy and j == slient_time:
                    pygame.draw.rect(screen, (255,0,0), pygame.Rect(i * 100, j * 100, 100, 100))
        pygame.display.flip()

        # convert image so it can be displayed in OpenCV
        view = pygame.surfarray.array3d(screen)

        #  convert from (width, height, channel) to (height, width, channel)
        view = view.transpose([1, 0, 2])

        #  convert from rgb to bgr
        img = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        for i in range(Q1.shape[0]):
            for j in range(Q2.shape[1]):
                img = cv2.putText(img, "%.2f " % diff[i, j], (i * 100 + 20, j * 100 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 1, cv2.LINE_AA)
                img = cv2.putText(img, "E = %d" % i, (i * 100, j * 100 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "T = %d" % j, (i * 100, j * 100 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "Q1 = %.2f" % Q1[i, j], (i * 100, j * 100 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 255), 1, cv2.LINE_AA)
                img = cv2.putText(img, "Q2 = %.2f" % Q2[i, j], (i * 100, j * 100 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 255), 1, cv2.LINE_AA)

                # img_bgr = cv2.putText(img_bgr, "%.3f" % value[i,j], (i*100,j*100+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # else:
        #  img = cv2.putText(img_bgr, "Wait", (i*100,j*100+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.imshow('Q DIffs agent{d}'.format(d=agent_num),img)
        img = img[0:Q1.shape[0] * 100, 0:Q1.shape[1] * 100]
        #path = os.path.join(path, 'agent{d}'.format(d=agent_num))
        cv2.imshow('Q_DIffs_agent{d}_{e}.jpg'.format(d=agent_num, e = iteration), img)
        #out = cv2.VideoWriter('Q_DIffs_agent{d}.avi'.format(d=agent_num), cv2.VideoWriter_fourcc(*'DIVX'),10,(Q1.shape[0] * 100, Q1.shape[1] * 100))
        #out.write(img)
        #out.release()


    def render_Q_diffs_video(self, Q1, Q2, agent_num,nummber_of_iterations):
        path = 'C:/Users/dvire/PycharmProjects/MA_QL/images/'
        path = os.path.join(path, 'agent{d}'.format(d=agent_num))
        out = cv2.VideoWriter('Q_DIffs_agent{d}.avi'.format(d=agent_num), cv2.VideoWriter_fourcc(*'DIVX'), 30,(Q1.shape[0] * 100, Q1.shape[1] * 100))
        for iter in range(nummber_of_iterations):
            if os.path.isfile(os.path.join(path , 'Q_DIffs_agent{d}_{i}.jpg'.format(d=agent_num, i=iter))):
                img = cv2.imread(os.path.join(path , 'Q_DIffs_agent{d}_{i}.jpg'.format(d=agent_num, i=iter)))
            #cv2.imshow('Q DIffs agent{d}'.format(d=agent_num),img)
                out.write(img)
        out.release()

    def render_q_by_agent(self, Qs, number_of_agents):
        plt.rc('legend', fontsize=4)
        iterations = Qs.shape[0]
        time_diffs = [[] for i in range(iterations)]
        labels = []
        lines = []
        fig2, axess = plt.subplots(number_of_agents , 1, figsize=(12, 5))
        for i in range(number_of_agents):
            for e in range(Qs.shape[4]):
                for s in range(Qs.shape[5]):
                    for c in range(Qs.shape[6]):
                        for j in range(iterations):
                            time_diffs[j] = Qs[j][0][i][0][e][s][c][0] - Qs[j][0][i][0][e][s][c][1]
                        axess[i].plot(range(iterations),time_diffs,linewidth=1)
                        labels.append('state ({e} {s} {c})'.format(e=e, s = s, c=c))
            axess[i].legend(labels, borderaxespad=0.)
            print(labels)
            labels = []
        #animator = ani.FuncAnimation(fig, chartfunc, interval=100)

#add legend

