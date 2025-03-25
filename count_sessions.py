"""Count the number of sessions for each animal and condition"""
import numpy as np
import pandas as pd
from utils import matchsession
from sessionlists import (dist60, dist90, dist120, TM20, TM10, TM2, TMrev2, TMrev10, TMrev20)

if __name__ == '__main__':

    animalList = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02', 
                'RatF30', 'RatF31', 'RatF32', 'RatF33', 'RatM30', 'RatM31', 'RatM32', 
                'RatF40', 'RatF41', 'RatF42', 'RatM40', 'RatM41', 'RatM42', 'RatM43', 
                    'RatF50', 'RatF51', 'RatF52', 'RatM50', 'RatM51', 'RatM52', 'RatM53', 'RatM54']

    results = np.zeros((len(animalList), 8))
    for i, animal in enumerate(animalList):
        for j, (sessionList, cond) in enumerate(zip([dist60, dist90, dist120, TM20, TM10, TM2+TMrev2, TMrev10, TMrev20], 
                                    ['dist60', 'dist90', 'dist120', 'TM20', 'TM10', 'TM2', 'TMrev10', 'TMrev20'])):
            results[i, j] = len(matchsession(animal, sessionList))
    df = pd.DataFrame(results, columns=['dist60', 'dist90', 'dist120', 'TM20', 'TM10', 'TM2', 'TMrev10', 'TMrev20'], index=animalList)
    average_sessions_by_rat = np.mean(results, axis=1)
    average_sessions_by_cond = np.mean(results, axis=0)

    total_sessions = np.sum(results)
    expected_sessions = len(animalList) * 6 * 8
    print(f'Total sessions: {total_sessions}')
    print(f'Expected sessions: {expected_sessions}')
    print(f'Expected sessions: {13 * 6 * 8}')
    print(f'Ratio sessions OK: {total_sessions/expected_sessions*100:.2f}%')
    print(f'Ratio sessions NOOK: {(expected_sessions-total_sessions)/expected_sessions*100:.2f}%')
    print(f'Average sessions by rat: {np.mean(average_sessions_by_rat):.2f}')

    print(df)