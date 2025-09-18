"""Count the number of sessions for each animal and condition"""
import numpy as np
import pandas as pd
from utils import matchsession
from sessionlists import (dist60, dist90, dist120, TM20, TM10, TM2, TMrev2, TMrev10, TMrev20)
from initialize_notebook import initialize_notebook

if __name__ == '__main__':

    root, rat_lists, rat_markers, (example_rat, example_session), GENERAL_PLOT_PARAMS = initialize_notebook()
    intact_rats, DSlesioned_rats, VSlesioned_rats, sham_rats, all_rats = rat_lists

    results = np.zeros((len(all_rats), 8))
    for i, animal in enumerate(all_rats):
        for j, (sessionList, cond) in enumerate(zip([dist60, dist90, dist120, TM20, TM10, TM2+TMrev2, TMrev10, TMrev20], 
                                    ['dist60', 'dist90', 'dist120', 'TM20', 'TM10', 'TM2', 'TMrev10', 'TMrev20'])):
            results[i, j] = len(matchsession(animal, sessionList))
    df = pd.DataFrame(results, columns=['dist60', 'dist90', 'dist120', 'TM20', 'TM10', 'TM2', 'TMrev10', 'TMrev20'], index=all_rats)
    average_sessions_by_rat = np.mean(results, axis=1)
    average_sessions_by_cond = np.mean(results, axis=0)

    total_sessions = np.sum(results)
    expected_sessions = len(all_rats) * 6 * 8
    print(f'Total sessions: {total_sessions}')
    print(f'Expected sessions: {expected_sessions}')
    print(f'Expected sessions: {13 * 6 * 8}')
    print(f'Ratio sessions OK: {total_sessions/expected_sessions*100:.2f}%')
    print(f'Ratio sessions NOOK: {(expected_sessions-total_sessions)/expected_sessions*100:.2f}%')
    print(f'Average sessions by rat: {np.mean(average_sessions_by_rat):.2f}')

    print(df)