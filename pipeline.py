import numpy as np
import pandas as pd
from glob import glob

def processer(folders, truncate = 1000, return_stim = True, mr_clean = True ) : 
    '''
    Args -
    folders : list of folders containing steinmetz data set
    truncate : length of time to grab from each mouse
    return_stim : returns a dataframe containing behavioral data
    ----------------
    Returns - 
    Neural : dataframe containing brain area for each neuron, spike times for that neuron, 
    and neuron ID
    Stimulus : dataframe containing mouse ID, stim onset time, response, contrast level, feedback
    '''
    neural_df = pd.DataFrame(columns = ['spike_t', 'area', 'maus'])
    if type(folders) == str : folders = [folders] #add for compatibility with pipeline iterating 
    for mouse, folder in enumerate(folders) :
        print(folder)
        spk = np.load(f'{folder}/spikes.times.npy', allow_pickle = True)
        spk = spk.reshape((spk.shape[0], ))
        if truncate != None : spk = spk[spk <= truncate]
            
        
        neurons = np.load(f'{folder}/spikes.clusters.npy', allow_pickle = True)
        neurons = neurons.reshape((neurons.shape[0], )) #array with neuron labels for each spike time
        if truncate != None : neurons = neurons[:spk.shape[0]] #cut to relevant subset

        loc = pd.read_csv(f'{folder}/channels.brainLocation.tsv', delim_whitespace = True)
        loc = loc['allen_ontology'].values #location of the ith channel

        channel = np.load(f'{folder}/clusters.peakChannel.npy', allow_pickle = True) - 1
        channel = channel.reshape(channel.shape[0], ).astype('int') #channel of the ith cluster
        #IDs mapping cluster - > channel. this is a clusterx1 array with each entry containing the channel ID of the ith cluster 
        print(f'Mapping neurons from {folder} to channels ... ')
        cluster_channel = [channel[cluster] for cluster in neurons] #generate an array containing the channel for each neuron
        #yields a list the same size as neurons with entries corresponding to the channel of that cluster
        print('Mapping channels to brain areas ...')
        brain_area = [loc[i] for i in cluster_channel] 
        #pull brain area 
        neural_subj_df = pd.DataFrame({'spike_t' : spk, 'neuron' : neurons, 'maus' : mouse, 
                                       'area' : brain_area}, index = np.arange(spk.shape[0]))
        neural_subj_df['maus'] = mouse
        neural_df = neural_df.append(neural_subj_df, ignore_index = True)
        
    if return_stim == True : 
        all_subj_df = pd.DataFrame(columns = ['cl', 'cr', 'stimtime', 'reward', 'response_type',
                                                 'rt'])
        print('''Getting behavioral data ...''')
        for i, folder in enumerate(folders) : 
            '''Grab all stimulus vectors. For the time paramaeters, we can use truncate to slice out
            the relevant indices. The shapes of these truncated versions can then be used to reshape
            categorical variables like contrast and reward type.'''
            rew_time = np.load(f'{folder}/trials.feedback_times.npy', allow_pickle = True)
            rew_time = rew_time.reshape(rew_time.shape[0], )
            #print(rew_time.shape)
            rew_time = rew_time[rew_time < truncate]
            #print(rew_time.shape)
            rew_type = np.load(f'{folder}/trials.feedbackType.npy', allow_pickle = True)
            rew_type = rew_type.reshape(rew_type.shape[0], )
            rew_type = rew_type[:rew_time.shape[0]]
                
            response_t =  np.load(f'{folder}/trials.response_times.npy', allow_pickle = True)
            response_t = response_t.reshape(response_t.shape[0], )
            response_t = response_t[response_t < truncate]
            response_type = np.load(f'{folder}/trials.response_choice.npy', allow_pickle = True)
            response_type = response_type.reshape(response_type.shape[0], )
            response_type = response_type[:response_t.shape[0]]
            
            stim = np.load(f'{folder}/trials.visualStim_times.npy', allow_pickle = True)
            stim = stim[stim < truncate]
            if stim.shape[0] == response_t.shape[0] + 1 : stim = stim[:-1]
            cl = np.load(f'{folder}/trials.visualStim_contrastLeft.npy', allow_pickle = True)
            cl = cl.reshape(cl.shape[0], )
            cl = cl[:stim.shape[0]]
            cr = np.load(f'{folder}/trials.visualStim_contrastRight.npy', allow_pickle = True)
            cr = cr.reshape(cr.shape[0], )
            cr = cr[:stim.shape[0]]
            
            response_t = response_t - stim
            
            #print(response_t.shape, response_type.shape, rew_time.shape, rew_type.shape,
                 #cl.shape, cr.shape, stim.shape)
            subj_df = pd.DataFrame({'cl' : cl, 'cr' : cr, 'stimtime' : stim, 'reward' : rew_type,
                            'response_type' : response_type, 'rt' : response_t})
            subj_df['maus'] = i
            all_subj_df = all_subj_df.append(subj_df, ignore_index = True)
            
        if mr_clean == True : all_subj_df = MR_clean(all_subj_df) 
            
        print('''~~~ OOOoOOOOooOoOOOOOOOOOOoooOooOOOOOOo ~~~''')
        
        return neural_df, all_subj_df
    
            
    else: return neural_df

def frequency_transform(folder, dat = None, truncate = 500, return_stim = True,
                        bin_size = 0.02, mr_clean = True) : 
    '''
    Args - 
    folders : glob folders to pass on to parent processer function
    bin_size : bin sizes for spike trains
    NOTE THAT THIS ONLY WORKS FOR ONE MOUSE. USE PIPELINE FOR MULTIPLE MICE. 
    ----------
    Returns - 
    Processer neural and stimulus dataframes as well as a frequency transformed version of 
    the neural df containign spike trains for each neuron over all trials
    '''
    if dat == None : 
        print('''Getting spike time data ...''')
        neur,  stim_df = processer(folder, truncate = truncate, return_stim = return_stim, mr_clean = mr_clean)
        stimuli = np.load(f'{folder}/trials.visualStim_times.npy', allow_pickle = True)
        
        cl = np.load(f'{folder}/trials.visualStim_contrastLeft.npy', allow_pickle = True)
        cl = cl.reshape(cl.shape[0], )
        cr = np.load(f'{folder}/trials.visualStim_contrastRight.npy', allow_pickle = True)
        cr = cr.reshape(cr.shape[0], )
        
        rew_time = np.load(f'{folder}/trials.feedback_times.npy', allow_pickle = True)
        rew_time = rew_time.reshape(rew_time.shape[0], )
        rew_type = np.load(f'{folder}/trials.feedbackType.npy', allow_pickle = True)
        rew_type = rew_type.reshape(rew_type.shape[0], )
        
        response_t =  np.load(f'{folder}/trials.response_times.npy', allow_pickle = True)
        response_t = response_t.reshape(response_t.shape[0], )
        response_type = np.load(f'{folder}/trials.response_choice.npy', allow_pickle = True)
        response_type = response_type.reshape(response_type.shape[0], )
    
    if truncate != None : 
        #truncate the data and then form a spike train with a fixed bin size over that interval.
        #use the histogram function with prefixed bins and range. slice out first element containing values.
        bins = np.arange(0, truncate + bin_size, bin_size) # i think we should delete + bin_size
        print('Forming groups ... ')
        dat = neur.loc[neur.spike_t < truncate] #slice time stamps before truncate time
        groups = [dat[dat.neuron == i].spike_t.values for i in dat.neuron.unique()]
        
        print('''Forming spike trains ...''')
        group_neur = [np.histogram(group, bins = bins, range = (0, truncate))[0] 
                  for group in groups]
        
        print('''Forming stimulus series ...''')
        stimuli = stimuli[stimuli < truncate] #we may need to change this to match processer syntax -- i.e. loading the stim shape based on rew time
        response_t = response_t[response_t < truncate]
        response_type = response_type[:response_t.shape[0]]
        response_type[response_type == 0] = 999
        if stimuli.shape[0] == response_t.shape[0] + 1 : stimuli = stimuli[:-1]
        group_stim = np.histogram(stimuli, bins = bins,  range = (0, truncate))[0]
        group_response = np.histogram(response_t, bins = bins,  range = (0, truncate))[0]
        group_response[group_response != 0] = response_type
        cr = cr[ : stimuli.shape[0]]
        cl = cl[ : stimuli.shape[0]]
        
        
        group_cr = np.zeros((int(truncate / bin_size)))
        group_cl = np.zeros((int(truncate / bin_size)))
        group_cr[group_stim != 0 ] = cr
        group_cl[group_stim != 0 ] = cl
        
        rew_time = rew_time[rew_time < truncate]
        group_reward = np.histogram(rew_time, bins = bins,  range = (0, truncate))[0]
        n_feedbacks = len(group_reward[group_reward != 0]) #use to slice out feed back times array
        rew_type = rew_type[:n_feedbacks]
        group_reward[group_reward != 0] = rew_type #each time a reward was delivered, assign -1 V 1
        #possible error here. 
        
    else : print('feed me truncate or grab code at top')

    
    print('Creating data-frame ...')
    index = np.arange(len(group_neur[0]))
    df = pd.DataFrame({ f'neuron_{i}' : spikes for i, spikes in enumerate(group_neur)},
                    index = index) #we could even include barea with neuron{i} for zip(dat.barea, group_neur)
    df['stim'] = group_stim
    df['contrast_left'] = group_cl
    df['contrast_right'] = group_cr
    df['response'] = group_response
    df['reward'] = group_reward
    print('~~~ the power of christ literally compels you ~~~')
    #return {'spikes' : df, 'full' : dat, 'behavioral' : stim_df} 
    return df, neur, stim_df 
'''spike trains, labelled neurons with spike time stamps, stimulus df'''

def pipeline(folders, path = None, truncate = 1000, bin_size = 0.02, mr_clean = True) : 
    '''
    Args - 
    Path : file path for all Steinmetz mice folders for loading. 
    Folders : list containing the paths of folders. Set to none if feeding path.
    Truncate : how long to truncate each mouse run . 
    -----------------
    Returns -
    frequency transform dataframes stitched together for all mice in folders, stimulus dataframe for all mice passed, time stamp neural spike data for all mice
    '''
    all_dat_neur = pd.DataFrame()
    all_dat_time_stamp_neur = pd.DataFrame()
    all_dat_stim = pd.DataFrame()
    #labels = []
    if path != None : folders = glob(path)
    for i, folder in enumerate(folders): 
        spike_df, labelled_neurons, stim_df = frequency_transform(folder, truncate = truncate, bin_size = bin_size, mr_clean = mr_clean)
        #label = [labelled_neurons.neuron, labelled_neurons.area]
        spike_df['maus'] = i 
        stim_df['maus'] = i
        labelled_neurons['maus'] = i
        all_dat_neur = all_dat_neur.append(spike_df, ignore_index = True)
        all_dat_stim = all_dat_stim.append(stim_df, ignore_index = True)
        all_dat_time_stamp_neur = all_dat_time_stamp_neur.append(labelled_neurons, ignore_index = True)
    return all_dat_neur, all_dat_stim, all_dat_time_stamp_neur     

def filt(folders, regions_to_pull = np.array(['GPe', 'LH', 'MB', 'MOp', 'MOs', 'MRN', 'TH'])) : 
    mice = pd.DataFrame(columns = ['Origin', 'abbrev'])
    '''
    pass path and relevant regions to pull. 
    Returns - file paths for folders containing areas of interest
    '''
    for i, folder in enumerate(folders) :
        #print(f'{folder}/channels.brainLocation.tsv' in glob(f'{folder}/*'))
        if f'{folder}/channels.brainLocation.tsv' in glob(f'{folder}/*') : 
            loc = pd.read_csv(f'{folder}/channels.brainLocation.tsv', delim_whitespace = True)
            loc = loc['allen_ontology'].values #loc ith channel
            n_relevant = 0
            add = False
            for area in regions_to_pull : 
                if np.any(loc == area) : 
                    n_relevant += 1
            df = pd.DataFrame({'Origin' : folder, 'abbrev' : folder[-20:],
                                      'n_relevant_probes' : n_relevant}, index = np.array([0]))
            mice = mice.append(df, ignore_index = True)
                    
    return mice

def stimulus(row) : 
    '''return 1 for left, 0 for right contrast. corresponds to split_by column '''
    if row.cl > row.cr : return 1
    elif row.cr > row.cl : return 0
    elif row.cr == row.cl : return '999'
    
def reaction_coding(row) : 
    '''response column'''
    if row.response_type == 0 : return '999'
    if row.response_type == -1 : return 0
    else : return 1
    
def MR_clean(dat) : 
    '''prepares data for hddm.'''
    dat = dat[dat.cl != dat.cr] # no equal contrast
    dat = dat[dat.response_type != 0] # no no-gos
    dat['feedback'] = dat.reward.apply(lambda x : 0 if x == -1. else x) #turn -1 to 0
    dat['split_by'] = dat.apply(stimulus, axis = 1)
    dat['response'] = dat.apply(reaction_coding, axis = 1 )
    dat['q_init'] = 0.7
    dat = dat.rename(columns = {'maus' : 'subj_idx'})
    dat = dat.loc[dat.split_by != 999]
    dat = dat.loc[dat.response != 999]
    dat = dat.loc[: , ['rt', 'subj_idx', 'response', 'response_type',
                       'split_by', 'feedback', 'q_init', 'stimtime']]
    return dat

def extract_trial_stim_avg(zone, stim, relevant_n, bins = .20) : 
    ''' 
    zone : processer type df containing maus, area, neuron, and spike time 
    relevant : neurons to slice out
    stimulus_df : MR_clean df containing stim times 
    -------------------
    returns : df containing spike averages for a given time following stimulus onset'''
    full_df = pd.DataFrame()
    for mouse in zone.maus.unique() : 
        print(f'Generating mouse {mouse} spike rate data ...')
        mouse_df = pd.DataFrame()
        relevant_n_slice = zone.loc[(zone.maus == mouse) & 
                                    (zone.area.isin(relevant_n))] #grab any relevant neuron subset
        for area in relevant_n_slice.area.unique() :
            # grab a slice of the data containing a particular relevant area 
            area_subset = relevant_n_slice[relevant_n_slice.area == area]
            area_spike_set = np.array([])    
            num_neurons = len(area_subset.neuron.unique())
            for stim_time in stim.loc[stim.subj_idx == mouse].stimtime :
                #calculate trial by trial post stimulus spiking avg 
                n_spikes = len(
                    area_subset[(area_subset.spike_t <= stim_time + bins) &
                               (stim_time <= area_subset.spike_t)].spike_t.values)
                trial_avg = n_spikes / bins / num_neurons
                area_spike_set = np.append(area_spike_set, trial_avg)
            mouse_df[area] = area_spike_set
            #print(area_spike_set.shape)
        mouse_df['maus'] = mouse 
        full_df = full_df.append(mouse_df, ignore_index = True)
    print('~~~ young success ~~~~')
    return full_df 
