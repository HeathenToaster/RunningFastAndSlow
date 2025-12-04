import glob
import fnmatch
from plotting import (create_teal_palette, create_blue_violet_palette,
                      create_kiwi_green_palette, create_red_palette)

# Define root for behavioral data 'path/to/DATA'
root = "DATA" # simlink: ln -s /home/thomas/Desktop/ALLDATA DATA

def initialize_notebook(root=root, ):

    print(f'Found {len(glob.glob(root+"/R*"))} rats in the {root} folder')

    example_rat = 'RatM00'
    example_session = 'RatM00_2021_07_22_16_13_03'

    # Define rat lists
    intact_rats = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02',
                   'RatF32', 'RatF33', 'RatM31', 'RatM32', 'RatF42', 'RatM40', 'RatM43',
                   'RatM53', 'RatM54', 'Raz41F', 'Raz42F', 'Raz46M', 'Raz47M'
                   ] 
    DSlesioned_rats = ['RatF30', 'RatF31', 'RatM30', 'RatF40', 'RatF41', 'RatM41', 'RatM42',
                       'RatF50', 'RatF51', 'RatF52', 'RatM50', 'RatM51', 'RatM52',
                    #    'Raz52F',  # Not doing task
                    #    'Raz53M',  # Weird fits, but could be added
                       'Raz55M',
                       'Raz63F', 'Raz64F', 'Raz65M', 'Raz66M', 'Raz67M',
                       ]
    VSlesioned_rats = [
        'Raz10F', 'Raz11F', 'Raz13F', 'Raz16M',
                       'Raz30F', 'Raz31F','Raz33M', 'Raz34M', 'Raz35M',
                       'Raz40F',  'Raz44M', 'Raz45M', #'Raz43M',
                       'Raz56M', 'Raz57M',
                        'Raz62F', #'Raz61F',
                       ]
    sham_rats = [
        'Raz14F', 'Raz19M', 'Raz32F', 'Raz36M'
                 ]
    all_rats = intact_rats + DSlesioned_rats + VSlesioned_rats + sham_rats
    rat_lists = intact_rats, DSlesioned_rats, VSlesioned_rats, sham_rats, all_rats

    male_list = [rat for rat in intact_rats if 'M' in rat]
    female_list = [rat for rat in intact_rats if 'F' in rat]

    print(f'Listed {len(all_rats)} rats in total, of which {len(intact_rats)} are intact ({len(female_list)}♀, {len(male_list)}♂), '
          f'{len(DSlesioned_rats)} DS lesions, {len(VSlesioned_rats)} VS lesions and {len(sham_rats)} shams.')
    print(f'Example rat is {example_rat}, example session: {example_session}')

    # Define marker and color for each rat, used in plots
    rat_markers = {}
    male_palette = create_teal_palette(num_shades=len(male_list))
    female_palette = create_blue_violet_palette(num_shades=len(female_list), start_intensity=80)
    ds_palette = create_kiwi_green_palette(num_shades=len(DSlesioned_rats))
    vs_palette = create_red_palette(num_shades=len(VSlesioned_rats))
    sham_palette = [(.5, .5, .5, .5) for _ in sham_rats]

    for (rat_list, palette) in zip([male_list, female_list, DSlesioned_rats, VSlesioned_rats, sham_rats],
                               [male_palette, female_palette, ds_palette, vs_palette, sham_palette]):
        for (rat, color) in zip(rat_list, palette):
            rat_markers[rat] = [color, 'o', '-']

    GENERAL_PLOT_PARAMS = {'alpha_0':  dict(yticks=[0, 1, 2, 3, 4, 5], yticksdiff=[-0.6, -0.3, 0, 0.3, 0.6], ylabel=r'$\alpha_0$'),
                           'alpha_t':  dict(yticks=[-.2, 0, .2, .4, .6], yticksdiff=[-.2, -0.15, -0.1, -0.05, 0, 0.05, .1, 0.15], ylabel=r'$\alpha_t$'),
                           'alpha_u':  dict(yticks=[-1.5, -1, -0.5, 0, .5], yticksdiff=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], ylabel=r'$\alpha_u$'),
                           'gamma_0':  dict(yticks=[0, 0.5, 1, 1.5, 2], yticksdiff=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], ylabel=r'$\gamma_0$'),
                           'gamma_t':  dict(yticks=[-.2, -.1, 0, .1], yticksdiff=[-0.04, -.02, 0, 0.02, 0.04, 0.06], ylabel=r'$\gamma_t$'),
                           'gamma_u':  dict(yticks=[-.6, -.4, -.2, 0, .2], yticksdiff=[-.2, -.1, 0, .1, .2], ylabel=r'$\gamma_u$'),
                           'mu_0':     dict(yticks=[0, 1, 2, 3, 4], yticksdiff=[-0.4, -.2, 0, 0.2, 0.4], ylabel=r'$\mu_0$'),
                           'mu_t':     dict(yticks=[-0.1, 0, .1, .2], yticksdiff=[-0.06, -0.04, -0.02, 0, 0.02, 0.04], ylabel=r'$\mu_t$'),
                           'mu_u':     dict(yticks=[-0.3, -0.2, -.1, 0, .1, .2],yticksdiff=[-0.06, -0.03, 0, 0.03, 0.06], ylabel=r'$\mu_u$'),
                           'sigma_0':  dict(yticks=[0, .1, .2, .3, .4, .5], yticksdiff=[-0.06, -0.03, 0, 0.03, 0.06], ylabel=r'$\sigma_0$'),
                           'sigma_t':  dict(yticks=[-.06, -.03, 0, .03, .06, 0.09], yticksdiff=[-0.02, -0.01, 0, 0.01, 0.02], ylabel=r'$\sigma_t$'),
                           'sigma_u':  dict(yticks=[-.05, 0, .05, .1, .15, .2], yticksdiff=[-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03], ylabel=r'$\sigma_u$'),
                           'effort_sensitivity_0':  dict(yticks=[0, 25, 50, 75, 100, 125], ylabel=r'$\varsigma_0$'),
                           'effort_sensitivity_t':  dict(yticks=[-10, 0, 10, 20, 30, 40], ylabel=r'$\varsigma_t$'),
                           'effort_sensitivity_u':  dict(yticks=[-10, -5, 0, 5, 10, 15, 20], ylabel=r'$\varsigma_u$'),
                           'beta':  dict(yticks=[0, 1, 2, 3, 4], ylabel=r'$\beta$'),
                           'total_effort': dict(yticks=[0, 50, 100, 150, 200, 250], ylabel=r'$E_{TOT}$'),
                           'total_drops': dict(yticks=[200, 400, 600, 800], ylabel=r'$R_{TOT}$')}

    return root, rat_lists, rat_markers, (example_rat, example_session), GENERAL_PLOT_PARAMS

"""
root, rat_lists, rat_markers, (example_rat, example_session), GENERAL_PLOT_PARAMS = initialize_notebook()
intact_rats, DSlesioned_rats, VSlesioned_rats, sham_rats, all_rats = rat_lists
"""


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root, rat_lists, rat_markers, (example_rat, example_session), GENERAL_PLOT_PARAMS = initialize_notebook()
    intact_rats, DSlesioned_rats, VSlesioned_rats, sham_rats, all_rats = rat_lists

    fig, ax = plt.subplots(1, 1)
    for x, rat_list in enumerate([intact_rats, DSlesioned_rats, VSlesioned_rats, sham_rats]):
        for y, rat in enumerate(rat_list):
            color = rat_markers[rat][0]
            ax.scatter(x, y, s=100, color=color)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Intact', 'DS lesion', 'VS lesion', 'Sham'])
    ax.set_ylabel('Individual rats')
    ax.yaxis.set_visible(False)

    plt.savefig('color.png')


