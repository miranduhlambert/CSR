%Import Time and Temperature Data
%Date: 9-05-2022 to 9-05-2024
tisu=readmatrix("tisu_data_avg.txt");
%Sort the data into vectors
temperature=tisu(:,3);
time=tisu(:,2);

% %Plot Temperature Versus time
figure;
plot(time, temperature);
xlabel('Time');
ylabel('Temperature');
title('Tisu Temperature over Time');

% Check if time and temperature vectors have the same length
if length(time) ~= length(temperature)
    error('Time and Temperature vectors must have the same length.');
end

% Calculate Sampling Frequency (Fs)
Fs = 1 / mean(diff(time)); % Assumes uniform sampling in time vector

% Perform STFT on the data
% Sectioning chunks of data 1200 points per orbit revolution
windowLength = 18001*5;
overlapLength = 67504;
FFTLength = 18001*5;

% Perform STFT using temperature as the signal
[s, f, t] = stft(temperature, Fs, Window=hann(windowLength),OverlapLength=overlapLength, FFTLength=FFTLength);

%TO SEE THE CONTENTS OF THE S magnitude
% % Display the first few rows and columns of `s` (STFT matrix)
% disp('First few entries of s (STFT output):');
% disp(s(1:5, 1:5));  % Adjust indices as needed to view more rows or columns
% 
% % Display the first few frequency values `f`
% disp('First few frequency values (f):');
% disp(f(1:5));  % First 5 frequency bins
% 
% % Display the first few time values `t`
% disp('First few time values (t):');
% disp(t(1:5));  % First 5 time segments

% Read the BetaPrime Data
betaPrimeData = readmatrix('2022_2024_09_05_Betaprime.txt');

%Load the BetaPrime Values into a Vector
betaPrime=betaPrimeData(:,4);

% Start date for BetaPrime data, assuming 1 value per day
startDate_beta = datetime(2022, 9, 5);

% Create a daily time vector for 731 days (2 years)
t_betaPrime = startDate_beta + days(0:730); % 731 entries, 0 to 730 days

% Convert Temperature time data to datetime for plotting clarity
startDate_temp = datetime(2022, 9, 5); % Starting date for temperature data
t_temperature = startDate_temp + seconds(t); % Ensure 't' is in seconds

% Interpolate BetaPrime to match t_temperature
betaPrime_interp = interp1(t_betaPrime, betaPrime, t_temperature, 'linear', 'extrap');

% Calculate the log-magnitude for plotting
log_magnitude = log(abs(s) / windowLength);

% Plot the log magnitude data
figure;
surf(t_temperature, f * 86400, log_magnitude, 'EdgeColor', 'none');
colormap(jet);
colorbar;
axis tight;
view(2);
ylim([0,35])

xlabel('Date');
ylabel('Frequency (cycles/day)');
title('STFT of Tisu Signal (Log Magnitude) with Celsius Labels');

% Set the color axis limits based on the range of log_magnitude values
caxis([min(log_magnitude(:)), max(log_magnitude(:))]);

% Custom Colorbar to Display Celsius Ticks
cb = colorbar;

% Define the Celsius values you want as tick marks
celsius_ticks = [min(abs(s(:))/windowLength),1e-5,0.001,0.01,1,max(abs(s(:))/windowLength)];  % Adjust based on your actual Celsius range

% Convert these Celsius values to their corresponding log scale values
log_ticks = log(celsius_ticks);

% Set the colorbar ticks to the log values
cb.Ticks = log_ticks;
cb.TickLabels = string(celsius_ticks) + " °C"; % Label ticks as Celsius

% Overlay the BetaPrime data on a separate right y-axis
hold on;
yyaxis right;
plot(t_temperature, betaPrime_interp, 'w--', 'LineWidth', 2); % Interpolated BetaPrime plot
ylabel('BetaPrime');

% Customize legend and color contrast settings
legend('STFT Log Magnitude', 'Interpolated BetaPrime', 'Location', 'Best');
set(gca, 'YColor', 'k'); % Set right y-axis color for BetaPrime

% Adjust the x-axis tick format to show dates in a readable format
xtickformat('dd-MMM-yyyy HH:mm:ss');
ax = gca;
ax.XTick = linspace(t_temperature(1), t_temperature(end), 15); % 15 evenly spaced ticks
ax.XTickLabelRotation = 45; % Rotate labels for readability

% Debugging steps for f, s, and t_temperature
disp('Size of f:');
disp(size(f));
disp('Size of s:');
disp(size(s));
disp('Size of t_temperature:');
disp(size(t_temperature));

% Define the range of frequencies you're interested in
min_freq = 15.4;  % Example minimum frequency (in cycles per day)
max_freq = 15.5;  % Example maximum frequency (in cycles per day)

% Find the indices of the frequencies that fall within the desired range
freq_indices = find((f*86400) >= min_freq & (f*86400) <= max_freq);

% Check if the frequency range is correct (debugging step)
disp('Selected frequency indices:');
disp(freq_indices);

% Display the corresponding frequencies
if ~isempty(freq_indices)
    disp('Corresponding frequencies:');
    disp(f(freq_indices));
end

% Initialize an empty cell array to store the data (time, frequency, magnitude)
selected_data = {};  

% Loop through each time point (t_temperature)
for time_idx = 1:length(t_temperature)
    disp(['Processing time index: ', num2str(time_idx)]);  % Debugging the time loop
    
    % Loop through each frequency index in the specified range
    for i = 1:length(freq_indices)
        freq_idx = freq_indices(i);  % Actual frequency index
        
        % Get the absolute magnitude normalized by the window length
        magnitude = abs(s(freq_idx, time_idx))*2/windowLength;
        
        % Convert the datetime to a string for concatenation
        time_str = datestr(t_temperature(time_idx), 'dd-mmm-yyyy HH:MM:SS');
        
        % Append the date (as a string), frequency, and magnitude to the data array
        selected_data{end + 1, 1} = time_str;        % Time as string
        selected_data{end, 2} = f(freq_idx)*86400;         % Frequency
        selected_data{end, 3} = magnitude;           % Magnitude
    end
end

% Check if any data has been collected (debugging step)
disp('Number of data points collected:');
disp(length(selected_data));

% Write the data to a CSV file if available
if ~isempty(selected_data)
    writecell(selected_data, 'frequency_magnitude_data.csv');
    disp('Frequency, time, and magnitude data have been written to frequency_magnitude_data.csv');
else
    disp('No data collected. Please check the input values.');
end