%% Test script for running CBP spike sorting algorithm
%
% Original code by Chaitanya Ekanadham, 12/26/2012
% Lab for Computational Vision, NYU & HHMI
% http://www.cns.nyu.edu/~lcv/
%
% Updated by Peter H. Li, fall/winter 2013
% Changes/adjustments made by Eero, spring/fall 2014
%
% This script demonstrates a method for sorting spikes recorded on
% one or more extracellular electrodes.  Unlike the commonly used
% clustering methods, this can correctly recover temporally
% overlapping spikes by relying on a sparse inverse method known as
% Continuous Basis Pursuit (CBP).
%
% The spike sorting method is described in this publication:
%
% A unified framework and method for automatic neural spike identification.
% C Ekanadham, D Tranchina, and E P Simoncelli. J. Neuroscience Methods, 
% vol. 222, pp. 47--55, Jan 2014.
% doi: 10.1016/j.jneumeth.2013.10.001  
% http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Ekanadham13
%
% and the Continuous Basis Pursuit method is described here:
%
% Recovery of sparse translation-invariant signals with continuous basis pursuit.
% C Ekanadham, D Tranchina and E P Simoncelli. IEEE Trans. Signal Processing, 
% vol.59, num.10, pp. 4735-4744, Oct 2011. doi:10.1109/TSP.2011.2160058
% http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Ekanadham11

%% -----------------------------------------------------------------
% Setup
% Set the current working directory to the demo directory:
% cd spikesort_demo

% Run the setup function, which will print warnings or errors if there
% are issues detected (for example, mex/C files that need to be
% compiled for your system).

spikesort_demo_setup(pwd());

%% -----------------------------------------------------------------
% Load raw electrode data

% Data are assumed to be in a matlab (.mat) file, containing:
%   data: channel x time matrix of voltage traces
%   dt : the temporal sampling interval (in seconds)

% Here are two example datasets.  There are additional options in the
% example_data directory.
switch 0 
    case 0
        % Quiroga simulated data examples
        data_filename = 'C_Easy1_noise015.mat';
        
    case 1
        % Harris data example
        %     data_filename = 'example_data/harris_d533101_v2.mat';
        data_filename = 'harris_d533101.mat';
end

% Load data
data = load(data_filename, 'data', 'dt');
data.filename = data_filename;
clear data_filename;
if (data.dt > 1/8000)
    warning('Sampling rate is less than recommended minimum of 8kHz');
end

% Load global default parameters
params = load_default_parameters();

% Adjust to specific values appropriate for the current data set:
[data, params] = load_data_defaults(data, params);

% Plot a bit of raw data, to make sure it looks as expected:
if (params.general.plot_diagnostics)
    plotDur = min(3000,size(data.data,2));   %**TODO: magic number:
    Tstart = round((size(data.data,2)-plotDur)/2);
    inds = Tstart+[1:plotDur];

    figure(1); clf; subplot(2,1,1); 
    plot([inds(1), inds(end)]*data.dt, [0 0], 'k'); 
    hold on; plot((inds-1)*data.dt, data.data(:,inds)'); hold off
    axis tight; xlabel('time (sec)');  ylabel('voltage'); 
    title(sprintf('Partial raw data, nChannels=%d, dt=%.2fmsec', size(data.data,1), 1000*data.dt));

%**TODO: indicate filtering frequency range
    figure(2); clf; subplot(2,1,1);
    dftMag = abs(fft(data.data,[],2));
    if (size(dftMag,1) > 1.5), dftMag = sqrt(mean(dftMag.^2)); end;
    maxDFTind = floor(size(data.data,2)/2);
    semilogy(([1:maxDFTind]-1)/(maxDFTind*data.dt*2), dftMag(1:maxDFTind));
    axis tight; xlabel('frequency (Hz)'); ylabel('amplitude');
    title('Fourier amplitude, averaged over channels');
end

%% -----------------------------------------------------------------
% Preprocessing Step 1: Temporally filter the voltage trace(s)

% Remove low and high frequencies - purpose is to enable crude removal
% of segments containg spikes via local amplitude thresholding, after
% which the background noise covariance can be estimated.
%   - pad with constant values (to avoid border effects when filtering)
%   - filter with band/highpass filter
%   - trim padding
%   - remove mean from each channel
%   - scale globally (across all channels) to be in range[-1, 1]
%
% The pre-filtering depends on a set of parameters, fields of params.filtering, 
% which need to be adjusted for each data set:
%   - type : type of filter for preprocessing. Currently supports
%            "fir1" and "butter"
%
%   - freq : range of frequencies (in Hz) for designing filter
%                     Set as [] to avoid any pre-filtering.
%
%   - order : order of the filter

filtdata = FilterData(data, params.filtering);
dataMag = sqrt(sum(filtdata.data .^ 2, 1));

if (params.general.plot_diagnostics)
    thresh = params.general.noise_threshold;
    minZoneLen = params.whitening.min_zone_len;
    if isempty(minZoneLen), minZoneLen = params.general.waveform_len/2; end
    noiseZones = GetNoiseZones(dataMag, thresh, minZoneLen);
                                                      
    % Plot filtered data, indicating noise regions that will be used in next
    % step for estimation of noise covariance:
    figure(1); subplot(2,1,2); 
    col = [0.9 0.8 1];
    hold on; 
    zonesL = cellfun(@(c) c(1), noiseZones);
    zonesR = cellfun(@(c) c(end), noiseZones);
    visibleInds = find(((inds(1) < zonesL) & (zonesL < inds(end))) |...
                       ((inds(1) < zonesR) & (zonesR < inds(end))));
    nh=fill(data.dt*[[1;1]*zonesL(visibleInds); [1;1]*zonesR(visibleInds)],...
             thresh*[-1;1;1;-1]*ones(1,length(visibleInds)), col,...
             'EdgeColor',col);
    plot([inds(1), inds(end)]*data.dt, [0 0], 'k'); 
    dh = plot((inds-1)*data.dt, filtdata.data(:,inds)');
    hold off; 
    set(gca, 'Xlim', ([inds(1),inds(end)]-1)*data.dt);
    legend('noise regions');
    title('Filtered data, with noise regions to be used for covariance estimation');

    figure(2); subplot(2,1,2);
    dftMag = abs(fft(filtdata.data,[],2));
    if (size(dftMag,1) > 1.5), dftMag = sqrt(mean(dftMag.^2)); end;
    semilogy(([1:maxDFTind]-1)/(maxDFTind*data.dt*2), dftMag(1:maxDFTind));
    axis tight; xlabel('frequency (Hz)'); ylabel('amplitude');
    title('Fourier amplitude of filtered data');
    
    % Plot histogram of magnitudes, indicating threshold
    figure(3); clf
    [N,X] = hist(dataMag, 50);  bar(X,N);
    hold on;
    th= plot(params.general.noise_threshold*[1 1], [0 0.9*max(N)], 'r');
    hold off;
    legend(th, 'threshold'); 
    xlabel('average voltage magnitude (all channels)'); 
    title('Histogram of signal RMS amplitude');

    %**TODO: overlay a chi2 distribution
    
    % At this point, all regions below threshold should look like noise. If
    % not, may need to go back and lower the threshold, or modify the filtering 
    % parameters.
end

%% -----------------------------------------------------------------
% Preprocessing Step 2: Estimate noise covariance and whiten data

% Whiten the noise (assuming space/time separability). This is not strictly
% needed by the CBP method, but it makes the L2-norm portion of the objective 
% into a simple sum of squares, greatly improving computational efficiency.
%   - estimate "noise zones" (regions of length min_zone_len or more 
%     samples whose Lp-norm are under noise_threshold).
%   - compute noise ACF from these zones
%   - whiten each channel in time by taking inverse matrix sqrt of ACF
%   - whiten across channels by left-multiplying each time slice 
%     by inverse matrix sqrt of the covariance matrix.
%
%   The whitening process depends on the following parameters:
%   - whiten_trace : binary, specifies whether or not to whiten w.r.t. 
%                  estimated noise covariance. 
% 
%   - noise_threshold : if whiten_trace is non-zero, this threshold is used
%                     to crudely separate noise from signal, for purposes 
%                     of estimating the noise covariance and designing the
%                     whitening filter.  The root-mean-squared signal value
%                     is compared to this threshold (taken across 
%                     electrodes).  

data_pp = PreprocessTrace(filtdata, params);
dataMag = sqrt(sum(data_pp.data .^ 2, 1));

% If run with plot set to 'true', the figures should help to verify/adjust 
% parameter settings: 
%**TODO: these need to be clarified and better
% aligned with the figures
%
% 1. In Figure 2, red zones are those with L2-norm less than the noise
% threshold, and should look like background activity.
% If not, adjust params.general.noise_threshold accordingly.
%
% 2. If original ACF is not going to zero, increase noise_pars.num_acf_lags.
% If original ACF is noisy, there may not be enough data samples to
% robustly estimate it. You can increase the amount of data in several ways:
%   a. increase params.general.noise_threshold ( to allow more samples)
%   b. decrease params.whitening.num_acf_lags
%   c. decrease params.whitening.min_zone_len (to allow shorter noise zones)
%
% 3. Spatial covariance matrix of whitened signal will be the identity matrix
% and ideally whitened ACF should be a delta function. In practice, this may
% not be the case because of (1) non-separability of noise in space/time,
% and/or (2) finite temporal window for estimating correlations. 
% If whitened temporal/spatial correlations are not white, try the following:
%   a. increase params.whitening.num_acf_lags
%   b. increase params.whitening.min_zone_len
% Note that this trades off with the quality the estimates (see #2 above).
%
% 4. In Figure 142, distribution of noise samples (blue) should be reasonably fit
%    by Generalized Gaussian distribution (red). We assume as default an
%    exponent of 2 (Gaussian). If this is not a good match, consider
%    changing the exponent accordingly (params.whitening.p_norm).

if (params.general.plot_diagnostics)
    thresh = mean(dataMag) + 3*std(dataMag);
    
    peakInds = dataMag(inds)>thresh;  
    peakLen = params.clustering.peak_len;
    if (isempty(peakLen)), peakLen=floor(params.general.waveform_len/2); end;
    for i = -peakLen : peakLen
        peakInds = peakInds & dataMag(inds) >= dataMag(inds+i);
    end
    peakInds = inds(peakInds);

    % Plot, over-writing the filtered data
    figure(1); subplot(2,1,2); cla
    col = [0.6 1 0.7];
    hold on;
    sh = patch(data.dt*[[1;1]*(peakInds-peakLen); [1;1]*(peakInds+peakLen)], ...
          thresh*[-1;1;1;-1]*ones(1,length(peakInds)), col,'EdgeColor',col);
    plot((inds-1)*data.dt, data_pp.data(:,inds)');
    hold off
    axis tight
    title('Filtered & whitened voltage traces')
    legend('Segments containing potential spikes');
    
    figure(2); subplot(2,1,2);
    dftMag = abs(fft(data_pp.data,[],2));
    if (size(dftMag,1) > 1.5), dftMag = sqrt(mean(dftMag.^2)); end;
    semilogy(([1:maxDFTind]-1)/(maxDFTind*data.dt*2), dftMag(1:maxDFTind));
    axis tight; xlabel('frequency (Hz)'); ylabel('amplitude');
    title('Fourier amplitude of filtered & whitened data');
    
    %**TODO: Replace with higher level wrapping call taking just data and
    % params.general
    wndata    = windowed_norm(data.data,    params.general.waveform_len);
    wndata_pp = windowed_norm(data_pp.data, params.general.waveform_len);
    figure(3);
    subplot(1,2,1);
    hist(wndata', 100);
    title('Windowed 2-Norm of original data');
    subplot(1,2,2);
    [N,X] = hist(wndata_pp', 100);   bar(X,N);
    hold on;
    plot(thresh*[1 1], [0 0.9*max(N)], 'r');
    hold off;

    title('Windowed 2-Norm of filtered and whitened data');
    clear wndata wndata_pp;

% At this point, potential spikes should be identified, and background 
% activity should look like univariate white noise.
end


%% -----------------------------------------------------------------
% Preprocessing step 3: Estimate initial spike waveforms

% Initialize spike waveforms:
%  - gather all data windows with L2-norm larger than threshold
%  - align spikes in these windows
%  - Compute PCA on these segments
%  - Perform K-means clustering in the subspace of the principal components
%    accounting for params.clustering.percent_variance portion of the total 
%    variance.
%
%  Parameters:
%  - num_waveforms : number of cells to be recovered from data.
%
%  - waveform_len : length (in samples) of spike waveforms.
%
%  - align_mode : During the clustering stage used to initialize spike
%                waveforms, segments of the trace are identified using a
%                threshold. These segments are then aligned based on this
%                parameter. Supported values are:
%                   maxrms : align wrt max L2 norm (RMS) value in segment
%                   maxabs : "" "" max L1 norm value in segment
%                   max : max signed sum across electrodes
%                   min : min "" "" "" 
%
%                For example, if the spike waveforms always have large
%                positive (negative) peaks, use max (min).

clear filtdata; % Recover some memory: don't need this any more

[centroids, assignments, X, XProj, PCs, snippet_centers_cl] = ...
    EstimateInitialWaveforms(data_pp, params);

% Plot the data segments projected into the space of the 2 leading 
% principal components.

VisualizeClustering(XProj, assignments, X, data_pp.nchan);
%**TODO: include background (noise) cluster in black

% For comparison to the CBP results, also grab the spike times
% corresponding to the segments assigned to each cluster:
spike_times_cl = GetSpikeTimesFromAssignments(snippet_centers_cl, assignments);

% At this point, waveforms of all potential cells should be identified.  If
% not, may need to adjust num_waveforms and re-run the clustering to 
% identify more/fewer cells.

% May also wish to adjust the waveform_len, increasing it if it seems the
% waveforms are being chopped off, or shortening it if there is a
% substantial boundary region of silence.  If you do this, you need to go
% back and re-run the fil

%% Step 4: chop signal into chunks for improved efficiency of CBP

% Partition data into snippets, separated by "noise zones" in which
% the RMS of the waveforms does not surpass "threshold" for at least
% "min_separation_len" consecutive samples. Snippets are constrained 
% to have duration between min/max_snippet_len.
%
% NOTE: This is for computational speedup only, so choose a  
% conservative (low) threshold to avoid missing spikes!

%**TODO: should use windowed_norm above?
% **TODO??: choose partion_pars.threshold based on sliding Lp-norm.
% The threshold should be the minimum window Lp-norm containing any overlap
% of any pair of waveforms. Display a histogram of sliding Lp-norms with
% this automated choice for the user.

data_rms = sqrt(sum(data_pp.data .^ 2, 1));% root-mean-squared across electrodes
%cluster_threshold = 4 * median(data_rms) ./ 0.6745; % robust
threshold = 4 * std(data_rms);

partition_pars = struct('threshold', threshold, ...
                        'smooth_len', 1, ...
                        'min_separation_len', ...
                        floor(params.general.waveform_len / 2), ...
                        'min_snippet_len', ...
                        params.general.waveform_len, ...
                        'max_snippet_len', ...
                        1001, ... % not enforced, only warnings
                        'min_pad_size', 5);
                    
[snippets, breaks, snippet_lens, snippet_centers, snippet_idx] = ...
    PartitionSignal(data_pp.data, partition_pars);
fprintf('Chopped up signal into %d chunks\n', length(snippets));

clear data_rms threshold

%**TODO: plot partitioned voltage trace (scrolling GUI)

% Histogram of windowed norm of snippets versus "silences"
wnsnip = cellfun(@(s) windowed_norm(s', params.general.waveform_len), snippets, 'UniformOutput', false);
wnbreak = windowed_norm(cell2mat(breaks')', params.general.waveform_len);
figure();
subplot(1,2,1);
hist(cell2mat(wnsnip')', 100);
title('Windowed 2-Norm of snippets');
subplot(1,2,2);
hist(wnbreak', 100);
title('Windowed 2-Norm of putative silences')


%% -----------------------------------------------------------------
% CBP setup
% Should be able to leave most of these defaults
num_waveforms = params.clustering.num_waveforms;

% The polar_1D version seemed to be cutting things down too much...
adjust_wfsize_fn = @(w) w; %polar_1D_adjust_wfsize(w, 0.1, 0.025, 301), ...

% cbp_outer_pars are parameters for learning the waveforms.
cbp_outer_pars = struct( ...
    'num_iterations', 2e2, ... % number of learning iterations
	'batch_size', 125, ... % batch size for learning
	'step_size', 5e-2, ... % step size for updating waveform shapes
	'step_size_decay_factor', 1, ... % annealing
	'plotevery',1, ... % plot interval
	'stop_on_increase', false, ... % stop when objective function increases
	'check_coeff_mtx', true, ... % sanity check (true to be safe)
	'adjust_wfsize', ... % called each iteration to adjust waveform size
        adjust_wfsize_fn, ...
	'rescale_flag', false, ... % always FALSE 
	'renormalize_features', false, ... % always FALSE
	'reestimate_priors', false, ... % always FALSE
    'CoeffMtx_fn', @polar_1D_sp_cnv_mtx, ... % convolves spikes w/waveforms
    'plot_every', 1 ... % plotting frequency    
);

% Set initial estimates of spike waveforms to the clustering centroids
cbp_outer_pars.init_features = cell(size(centroids, 2), 1);
cbp_outer_pars.num_chan = data_pp.nchan;
for i = 1 : num_waveforms
    cbp_outer_pars.init_features{i} = ...
        reshape(centroids(:, i), [], cbp_outer_pars.num_chan);
end 

% cbp_pars are parameters for doing sparse inference.
% ADD : incorporate estimates of prior firing rates from clustering result
% ADD : pnorm of noise (determined above)
cbp_pars = struct ( ...
    'noise_sigma',  data_pp.noise_sigma, ... % Optimization parameters
    'firing_rates', 1e-3 .* ones(num_waveforms, 1), ... % prior firing rate
    'cbp_core_fn', @polar_1D_cbp_core, ... % CBP core interpolation
    'solve_fn', @cbp_ecos_2norm, ... % Optimization solver function
    'debug_mode', false, ... % debug mode
    'num_reweights', 1e3, ... % MAX number of IRL1 iterations
    'magnitude_threshold', 1e-2, ... % amplitude threshold for deleting spikes
    'parfor_chunk_size', Inf, ... % parallelization chunk size
    'num_features', num_waveforms ... % number of "cells"
);


%% -----------------------------------------------------------------
% CBP parameters that should be adjusted by user

% For picking template delta
cbp_pars.accuracy = 0.1;

% Try single-spike soln first
cbp_pars.compare_greedy = false; 
cbp_pars.greedy_p_value = 0;
% cbp_pars.greedy_p_value = 1 - 1e-5; % tol. to accept initial greedy soln

% Corr. threshold below which atoms will not be used during CBP.
% For speedup only; set to 0 to disable
cbp_pars.prefilter_threshold = 0; %0.01, ... 


%% -----------------------------------------------------------------
% Pick solver and reweighting parameters

% Setup CVX if needed
% addpath(fullfile(sstpath, '../cvx/'));
% cvx_setup;
% cvx_solver sedumi; % Can also use ECOS if using CVX2 and ECOS shim installed.

cbp_pars.solve_fn = @cbp_ecos_2norm;
reweight_exp = 1.5 * [1 1 1];
% cbp_pars.solve_fn = @cbp_cvx;
% cbp_pars.solve_fn = @cbp_qcml_sumsquare;
% reweight_exp = 25 * [1 1 1];

cbp_pars.lambda = reweight_exp(:); % multiplier for sparsity weight

% FIXME: Move function definition inside CBP, but leave explanation of
% reweight_exp for users.
%
% Set the reweighting function for the Iteration Reweighted L1 optimization
% for inferring the spikes. Theoretically, the new weight for a coefficient
% x should be set to -d/dz(log(P(z)))|z=x where P(z) is the prior density
% on the spike amplitudes. Here we employ a power-law distribution for 
% 0 <= x <= M with exponent=reweight_exp and offset=eps
cbp_pars.reweight_fn = cell(num_waveforms, 1);
for i = 1 : num_waveforms
    cbp_pars.reweight_fn{i} = @(x) reweight_exp(i) ./ (eps + abs(x));    
end

%% -----------------------------------------------------------------
% Run CBP, estimating spike times of all cells
% matlabpool open
starttime = tic;
cbp_pars.progress = true; % Set false if having Java errors from progress bar
[spike_times, spike_amps, recon_snippets] = ...
    SpikesortCBP(snippets, ...
                 snippet_centers, ...
                 cbp_outer_pars, ...
                 cbp_pars);
toc(starttime);

% Postprocess params
% TODO: Just make this a default value, either 30 or 40 inside relevant
% functions.  But first have to decide whether to try to reshift spike
% times to remove consistent bias; tricky as depends on how ground truth is
% defined, e.g. simulation versus intracellular electrode.
spike_location_slack = 30; % For live updating ground truth feedback

%% -----------------------------------------------------------------
% Pick amplitude thresholds and visualize effect on ACorr/XCorr
% NB: Much faster if mex trialevents.c is compiled

% Clustering XCorr and ACorr plots
% Optional
% spike_times_cl = GetSpikeTimesFromAssignments(snippet_centers_cl, assignments);
% f = figure();
% % ACorrs
% for i = 1:NUM_WAVEFORMS
%     subplot(NUM_WAVEFORMS, NUM_WAVEFORMS, sub2ind([NUM_WAVEFORMS NUM_WAVEFORMS], i, NUM_WAVEFORMS));
%     psthacorr(spike_times_cl{i}.*dt)
% end
% 
% % XCorrs
% for i = 1:NUM_WAVEFORMS
%     for j = i+1 : NUM_WAVEFORMS
%         subplot(NUM_WAVEFORMS, NUM_WAVEFORMS, sub2ind([NUM_WAVEFORMS NUM_WAVEFORMS], j, i));
%         psthxcorr(spike_times_cl{i}.*dt, spike_times_cl{j}.*dt)
%     end
% end


% True spike XCorr and ACorr plots (TST defined after load_data above)
% Optional
% ntruth = length(true_sp);
% figure();
% % ACorrs
% for i = 1:ntruth
%     sp = true_sp{i};
%     if isempty(sp), continue; end
%     
%     subplot(ntruth, ntruth, sub2ind([ntruth ntruth], i, ntruth));
%     psthacorr(sp.*dt)
% end
% % XCorrs
% for i = 1:ntruth
%     spi = true_sp{i};
%     if isempty(spi), continue; end
% 
%     for j = i+1 : ntruth
%         spj = true_sp{j};
%         if isempty(spj), continue; end
% 
%         subplot(ntruth, ntruth, sub2ind([ntruth ntruth], j, i));
%         psthxcorr(spi.*dt, spj.*dt)
%     end
% end

[atgf amp_threshold] = AmplitudeThresholdGUI(spike_amps, spike_times, 'dt', data_pp.dt, 'location_slack', spike_location_slack);

%% -----------------------------------------------------------------
% Histogram of windowed norm for data, whitened data, residual

data_recon = cell(size(snippets));
for i = 1:numel(snippets)
    data_recon{i} = snippets{i} - recon_snippets{i};
end
wnresid = cellfun(@(s) windowed_norm(s', params.general.waveform_len), data_recon, 'UniformOutput', false);

figure();
sanesubplot(1, 3, {1 1});
hist(cell2mat(wnsnip')', 100);
title('Windowed 2-Norm of snippets')
sanesubplot(1, 3, {1 2});
hist(wnbreak', 100);
title('Windowed 2-Norm of putative silences')
sanesubplot(1, 3, {1 3});
hist(cell2mat(wnresid')', 100);
title('Windowed 2-Norm of snippet residuals after CBP')


%% -----------------------------------------------------------------
% Calculate new waveform estimates with interpolation (defaults to cubic spline)
nlrpoints = (params.general.waveform_len-1)/2;
waveforms = cell(size(spike_times));
for i = 1:numel(spike_times)
    % Have to shift spiketimes by 1 because existing code treats data as 1 indexed.
    sts = spike_times{i}(spike_times{i} > amp_threshold(i)) - 1;
    waveforms{i} = CalcSTA(data_pp.data', sts, [-nlrpoints nlrpoints]);
end

% Show diff with initial estimates
figure();
nc = ceil(sqrt(num_waveforms));
nr = ceil(num_waveforms / nc);
for i = 1:numel(waveforms)
    subplot(nr, nc, i);
    inith = plot(cbp_outer_pars.init_features{i}, 'b');
    hold on
    finalh = plot(waveforms{i}, 'r');

    err = norm(cbp_outer_pars.init_features{i} - waveforms{i});
    title(sprintf('Norm of diff over norm of final: %.2f', ...
        err / norm(waveforms{i})));
	legend([inith(1) finalh(1)], {'Initial', 'Final'});
end


%% -----------------------------------------------------------------
% Set init_features to new waveforms and go back up to rerun CBP with new waveforms estimates.
% Optional
% cbp_outer_pars.init_features = waveforms;


%% -----------------------------------------------------------------
% Info on true spikes available?
ground_truth = load(data_pp.filename, 'true_spike_times', 'true_spike_class', 'dt');
ground_truth.filename = data_pp.filename;

% Acceptable slack for considering two spikes a match.  In units of samples.
% Currently two-sided, but this should probably be changed.
ground_truth.spike_location_slack = 30;

if isfield(ground_truth, 'true_spike_times') && isfield(ground_truth, 'true_spike_class')
    % Reformat as 1-cellarr per cell of spike times.
    ground_truth.true_sp = GetSpikeTimesFromAssignments(ground_truth.true_spike_times, ground_truth.true_spike_class);
    
    % Reorder to match cell numbering from clustering.
    [ground_truth.true_sp, ...
     ground_truth.true_spike_times, ... 
     ground_truth.true_spike_class] = ReorderCells( ...
        ground_truth.true_sp, spike_times_cl, ground_truth.spike_location_slack);
end

%% -----------------------------------------------------------------
% Plot various snippet subpopulations
[est_matches true_matches] = GreedyMatchTimes(spike_times, ground_truth.true_sp, ground_truth.spike_location_slack);

% Complete misses (missed even at 0 threshold)
completemisses = cell(size(ground_truth.true_sp));
for i = 1:length(ground_truth.true_sp)
    completemisses{i} = ground_truth.true_sp{i}(true_matches{i} == 0);
end

% All FPs (with 0 threshold)
allfps = cell(size(spike_times));
for i = 1:length(spike_times)
    allfps{i} = spike_times{i}(est_matches{i} == 0);
end

% For example; more complicated selections of spikes like misses or FPs
% with given threshold can also be calculated
% TODO PHLI: automate this.
% desiredspiketimes = cell2mat(allfps);         % Pick one or the other...
desiredspiketimes = cell2mat(completemisses); % Pick one or the other...
% desiredspiketimes = cell2mat(spike_times);

% Now find and plot the relevant snippets
snipindices = FindSnippets(desiredspiketimes, snippet_centers, snippets); % snippets just given as shorthand for calculating widths
ScrollSnippets(snippets, snippet_centers, ...
    'snipindices',  unique(snipindices(snipindices > 0)),  ...
    'cbp',          spike_times,        ...
    'cbpamp',       spike_amps,         ...
... %     'cbpampthresh', amp_threshold,      ... % Could use amp_threshold if we used that to pick snippets...
    'clust',        spike_times_cl,     ...
... %     'recons',       recon_snippets,     ...
    'true',         ground_truth.true_sp);


%% -----------------------------------------------------------------
% Visualize true spike assignments in PC-space

if isfield(ground_truth, 'true_spike_class') && isfield(ground_truth, 'true_spike_times')
    cluster_pars = params.clustering;
    if isempty(cluster_pars.window_len), cluster_pars.window_len = params.general.waveform_len; end
    cluster_pars.align_mode = data_pp.polarity;

    Xstar = ConstructSnippetMatrix(data_pp.data, ground_truth.true_spike_times, cluster_pars);
    % Remove mean component and project onto leading PC's
    XProjstar = (Xstar - repmat(mean(X, 2), 1, size(Xstar, 2)))' * PCs;
    VisualizeClustering(XProjstar, ground_truth.true_spike_class, Xstar, size(data_pp, 1), ...
                        figure(8), figure(9), '.');    
    
    % Plot the putative spikes w.r.t. the 2 leading principal components.
    % PC's are computed across all (aligned) windows which pass the threshold
    % test. K-means clustering is performed using the PC's accounting for
    % cluster_pars.percent_variance portion of the total variance.
    figure(8); title('True clustering');
end

% ADD NEW% Plot the putative spikes w.r.t. the 2 leading principal components.
% PC's are computed across all (aligned) windows which pass the threshold 
% test. K-means clustering is performed using the PC's accounting for
% cluster_pars.percent_variance portion of the total variance.
% FIGURE OF HISTOGRAM OF WHITENEDE RMS SAMPLES cluster sthreshold
% ADD multiple PC plots (optional) 

% What to check:
%
% 1. Adjust cluster_threshold to properly separate background activity from
%   spike data (shold cleanly separate histogram in Fig 200).
%
% 2. Adjust NUM_WAVEFORMS so that separate clusters in Fig 6 are identified
%    with separate colors (i.e. waveforms in Fig 7 should all have distinct
%    shapes).

%% -----------------------------------------------------------------
% Get greedy spike matches and plot RoC-style
% NB: Much faster if mex greedymatchtimes.c is compiled
PlotCBPROC(spike_times, spike_amps, ground_truth.true_sp, ground_truth.spike_location_slack);


%% -----------------------------------------------------------------
% Evaluation (ONLY if ground truth is available)

if isfield(ground_truth, 'true_spike_times') && isfield(ground_truth, 'true_spike_class')
    % Since we already permuted ground truth to match clustering, this is true
    % by definition
    best_ordering_cl = 1:length(spike_times_cl);
    best_ordering = 1:length(spike_times);
    
    % Evaluate clustering sorting
    [total_misses_cl, total_false_positives_cl, misses_cl, ...
        false_positives_cl] = ...
        evaluate_sorting(spike_times_cl, ground_truth.true_sp, ground_truth.spike_location_slack);
    fprintf('Clust %s', SortingEvaluationStr(ground_truth.true_sp, spike_times_cl, total_misses_cl, total_false_positives_cl));
    
    % Evaluate CBP sorting
    [total_misses, total_false_positives, prune_est_times, misses, ...
        false_positives] = ...
        EvaluateSorting(spike_times, spike_amps, ground_truth.true_sp, 'threshold', amp_threshold, 'location_slack', spike_location_slack);
    fprintf('  CBP %s', SortingEvaluationStr(ground_truth.true_sp, prune_est_times, total_misses, total_false_positives));
    
end


%% -----------------------------------------------------------------
% Things to add

% * visualization of CBP spikes in PC-space alongside clustering
