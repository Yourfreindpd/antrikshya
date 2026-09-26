#include <Arduino.h>
#include "driver/i2s.h"
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>

#include "model_data.h"
#include "KwsAsrStreamer.h"

// This is the feature and model contract in train_kws_model.py.
static constexpr int SAMPLE_RATE = 16000;
static constexpr int CLIP_SAMPLES = 16000;
static constexpr int FRAME_LEN = 400;
static constexpr int FRAME_STEP = 320;
static constexpr int FFT_SIZE = 512;
static constexpr int NUM_FRAMES = 49;
static constexpr int NUM_MELS = 40;
static constexpr int NUM_MFCC = 13;

// Change these three pins to match the connected I2S microphone.
static constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
static constexpr int I2S_SCK_PIN = 14;
static constexpr int I2S_WS_PIN = 15;
static constexpr int I2S_SD_PIN = 32;

// Wi-Fi / asr.py connection. Move these to a git-ignored secrets.h if this
// sketch is going in version control.
static const char* WIFI_SSID = "cupnoodles";
static const char* WIFI_PASSWORD = "hotwater";
static const char* ASR_SERVER_HOST = "10.158.127.62";  // PC running asr_server.py
static const uint16_t ASR_SERVER_PORT = 8000;

static KwsAsrStreamer streamer;
static constexpr int WAKE_CLASS_INDEX = 0;  // index of "antriksya" in CLASS_NAMES
static constexpr float WAKE_THRESHOLD = 0.90f;
static bool wake_threshold_crossed(float probability) {
	static bool was_above_threshold = false;
	bool is_above_threshold = probability >= WAKE_THRESHOLD;
	bool crossed = is_above_threshold && !was_above_threshold;
	was_above_threshold = is_above_threshold;
	return crossed;
}

static int16_t* audio_buffer = nullptr;
static float* mfcc = nullptr;
static float hamming_window[FRAME_LEN];
static float (*mel_filters)[FFT_SIZE / 2 + 1] = nullptr;
static float dct_basis[NUM_MFCC][NUM_MELS];
static float* fft_real = nullptr;
static float* fft_imag = nullptr;
static float* power_spectrum = nullptr;
static float* mel_energy = nullptr;

// Two reusable workspaces hold the largest pair of adjacent tensors.
static constexpr int MAX_TENSOR_VALUES = 32 * 25 * 13;
static float* tensor_a = nullptr;
static float* tensor_b = nullptr;

static const char* const CLASS_NAMES[KWS_NUM_CLASSES] = {
	"antriksya", "unknown", "silence"
};

// Mirrors every debug line to Serial and to the asr.py web dashboard.
static void log_line(const char* fmt, ...) {
	char buffer[256];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);
	Serial.println(buffer);
	streamer.sendLog(buffer);
}

static float hz_to_mel(float hz) {
	return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
	return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void initialize_features() {
	for (int i = 0; i < FRAME_LEN; ++i) {
		hamming_window[i] = 0.54f - 0.46f * cosf(
			2.0f * PI * i / static_cast<float>(FRAME_LEN - 1));
	}

	float mel_min = hz_to_mel(300.0f);
	float mel_max = hz_to_mel(8000.0f);
	int bins[NUM_MELS + 2];
	for (int i = 0; i < NUM_MELS + 2; ++i) {
		float mel = mel_min + (mel_max - mel_min) * i / (NUM_MELS + 1.0f);
		bins[i] = static_cast<int>(floorf(
			(FFT_SIZE + 1) * mel_to_hz(mel) / SAMPLE_RATE));
	}

	memset(mel_filters, 0,
	       NUM_MELS * (FFT_SIZE / 2 + 1) * sizeof(float));
	for (int m = 0; m < NUM_MELS; ++m) {
		int left = bins[m];
		int center = bins[m + 1];
		int right = bins[m + 2];
		for (int f = left; f < center && f <= FFT_SIZE / 2; ++f) {
			mel_filters[m][f] = (f - left) / (center - left + 1e-8f);
		}
		for (int f = center; f < right && f <= FFT_SIZE / 2; ++f) {
			mel_filters[m][f] = (right - f) / (right - center + 1e-8f);
		}
	}

	for (int k = 0; k < NUM_MFCC; ++k) {
		float factor = k == 0 ? sqrtf(1.0f / NUM_MELS)
							  : sqrtf(2.0f / NUM_MELS);
		for (int m = 0; m < NUM_MELS; ++m) {
			dct_basis[k][m] = factor * cosf(
				PI * (m + 0.5f) * k / NUM_MELS);
		}
	}
}

static void fft_512() {
	for (int i = 0, j = 0; i < FFT_SIZE; ++i) {
		if (i < j) {
			float real = fft_real[i];
			fft_real[i] = fft_real[j];
			fft_real[j] = real;
			float imag = fft_imag[i];
			fft_imag[i] = fft_imag[j];
			fft_imag[j] = imag;
		}
		int bit = FFT_SIZE >> 1;
		while (j >= bit && bit > 0) {
			j -= bit;
			bit >>= 1;
		}
		j += bit;
	}

	for (int length = 2; length <= FFT_SIZE; length <<= 1) {
		float angle = -2.0f * PI / length;
		float step_real = cosf(angle);
		float step_imag = sinf(angle);
		for (int start = 0; start < FFT_SIZE; start += length) {
			float current_real = 1.0f;
			float current_imag = 0.0f;
			int half = length >> 1;
			for (int offset = 0; offset < half; ++offset) {
				int even = start + offset;
				int odd = even + half;
				float odd_real = fft_real[odd] * current_real -
								 fft_imag[odd] * current_imag;
				float odd_imag = fft_real[odd] * current_imag +
								 fft_imag[odd] * current_real;
				fft_real[odd] = fft_real[even] - odd_real;
				fft_imag[odd] = fft_imag[even] - odd_imag;
				fft_real[even] += odd_real;
				fft_imag[even] += odd_imag;
				float next_real = current_real * step_real -
								  current_imag * step_imag;
				current_imag = current_real * step_imag +
								current_imag * step_real;
				current_real = next_real;
			}
		}
	}
}

static bool extract_mfcc() {
	for (int frame = 0; frame < NUM_FRAMES; ++frame) {
		int start = frame * FRAME_STEP;
		for (int i = 0; i < FFT_SIZE; ++i) {
			fft_real[i] = 0.0f;
			fft_imag[i] = 0.0f;
		}
		for (int i = 0; i < FRAME_LEN; ++i) {
			float current = audio_buffer[start + i] / 32768.0f;
			float previous = start + i == 0
				? current
				: audio_buffer[start + i - 1] / 32768.0f;
			fft_real[i] = (current - 0.97f * previous) * hamming_window[i];
		}

		fft_512();
		for (int f = 0; f <= FFT_SIZE / 2; ++f) {
			power_spectrum[f] = (fft_real[f] * fft_real[f] +
								 fft_imag[f] * fft_imag[f]) / FFT_SIZE;
		}
		for (int m = 0; m < NUM_MELS; ++m) {
			float energy = 0.0f;
			for (int f = 0; f <= FFT_SIZE / 2; ++f) {
				energy += mel_filters[m][f] * power_spectrum[f];
			}
			mel_energy[m] = logf(fmaxf(energy, 1e-6f));
		}
		for (int k = 0; k < NUM_MFCC; ++k) {
			float coefficient = 0.0f;
			for (int m = 0; m < NUM_MELS; ++m) {
				coefficient += dct_basis[k][m] * mel_energy[m];
			}
			mfcc[frame * NUM_MFCC + k] = coefficient;
		}
	}

	const int feature_count = NUM_FRAMES * NUM_MFCC;
	float mean = 0.0f;
	for (int i = 0; i < feature_count; ++i) mean += mfcc[i];
	mean /= feature_count;
	float variance = 0.0f;
	for (int i = 0; i < feature_count; ++i) {
		float difference = mfcc[i] - mean;
		variance += difference * difference;
	}
	float standard_deviation = sqrtf(variance / feature_count) + 1e-6f;
	for (int i = 0; i < feature_count; ++i) {
		mfcc[i] = (mfcc[i] - mean) / standard_deviation;
	}
	return true;
}

static float relu(float value) {
	return value > 0.0f ? value : 0.0f;
}

static void standard_convolution(const float* input, float* output,
								 int input_channels, int input_height,
								 int input_width, int output_channels,
								 int output_height, int stride_height,
								 const int8_t* weights, float weight_scale,
								 const float* bias, int kernel_height,
								 int kernel_width) {
	for (int oc = 0; oc < output_channels; ++oc) {
		for (int oy = 0; oy < output_height; ++oy) {
			for (int ox = 0; ox < input_width; ++ox) {
				float sum = bias[oc];
				for (int ic = 0; ic < input_channels; ++ic) {
					for (int ky = 0; ky < kernel_height; ++ky) {
						int iy = oy * stride_height + ky - kernel_height / 2;
						if (iy < 0 || iy >= input_height) continue;
						for (int kx = 0; kx < kernel_width; ++kx) {
							int ix = ox + kx - kernel_width / 2;
							if (ix < 0 || ix >= input_width) continue;
							int input_index = (ic * input_height + iy) * input_width + ix;
							int weight_index = (((oc * input_channels + ic) *
												 kernel_height + ky) * kernel_width) + kx;
							sum += input[input_index] * weights[weight_index] * weight_scale;
						}
					}
				}
				output[(oc * output_height + oy) * input_width + ox] = relu(sum);
			}
		}
	}
}

static void depthwise_convolution(const float* input, float* output,
								  int channels, int input_height,
								  int output_height, int width,
								  int stride_height, const int8_t* weights,
								  float weight_scale) {
	for (int c = 0; c < channels; ++c) {
		for (int oy = 0; oy < output_height; ++oy) {
			for (int ox = 0; ox < width; ++ox) {
				float sum = 0.0f;
				for (int ky = 0; ky < 3; ++ky) {
					int iy = oy * stride_height + ky - 1;
					if (iy < 0 || iy >= input_height) continue;
					for (int kx = 0; kx < 3; ++kx) {
						int ix = ox + kx - 1;
						if (ix < 0 || ix >= width) continue;
						int input_index = (c * input_height + iy) * width + ix;
						int weight_index = (c * 3 + ky) * 3 + kx;
						sum += input[input_index] * weights[weight_index] * weight_scale;
					}
				}
				// PyTorch applies ReLU after the pointwise convolution, not here.
				output[(c * output_height + oy) * width + ox] = sum;
			}
		}
	}
}

static int run_model(float probabilities[KWS_NUM_CLASSES]) {
	standard_convolution(mfcc, tensor_a, 1, 49, 13, 16, 25, 2,
						 conv0_weights, conv0_scale, conv0_bias, 5, 3);
	depthwise_convolution(tensor_a, tensor_b, 16, 25, 25, 13, 1,
						  ds1_dw_weights, ds1_dw_scale);
	standard_convolution(tensor_b, tensor_a, 16, 25, 13, 32, 25, 1,
						 ds1_pw_weights, ds1_pw_scale, ds1_pw_bias, 1, 1);
	depthwise_convolution(tensor_a, tensor_b, 32, 25, 13, 13, 2,
						  ds2_dw_weights, ds2_dw_scale);
	standard_convolution(tensor_b, tensor_a, 32, 13, 13, 32, 13, 1,
						 ds2_pw_weights, ds2_pw_scale, ds2_pw_bias, 1, 1);
	depthwise_convolution(tensor_a, tensor_b, 32, 13, 13, 13, 1,
						  ds3_dw_weights, ds3_dw_scale);
	standard_convolution(tensor_b, tensor_a, 32, 13, 13, 48, 13, 1,
						 ds3_pw_weights, ds3_pw_scale, ds3_pw_bias, 1, 1);

	float logits[KWS_NUM_CLASSES];
	for (int c = 0; c < KWS_NUM_CLASSES; ++c) {
		float pooled = 0.0f;
		for (int i = 0; i < 13 * 13; ++i) pooled += tensor_a[c * 13 * 13 + i];
		pooled /= 169.0f;
		logits[c] = fc_bias[c];
		for (int i = 0; i < 48; ++i) {
			float channel_average = 0.0f;
			for (int p = 0; p < 169; ++p) channel_average += tensor_a[i * 169 + p];
			logits[c] += (channel_average / 169.0f) * fc_weights[c * 48 + i] * fc_scale;
		}
	}

	float largest = logits[0];
	for (int c = 1; c < KWS_NUM_CLASSES; ++c) largest = fmaxf(largest, logits[c]);
	float total = 0.0f;
	for (int c = 0; c < KWS_NUM_CLASSES; ++c) {
		probabilities[c] = expf(logits[c] - largest);
		total += probabilities[c];
	}
	int prediction = 0;
	for (int c = 0; c < KWS_NUM_CLASSES; ++c) {
		probabilities[c] /= total;
		if (probabilities[c] > probabilities[prediction]) prediction = c;
	}
	return prediction;
}

static void initialize_i2s() {
	i2s_config_t config = {};
	config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
	config.sample_rate = SAMPLE_RATE;
	config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
	config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
	config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
	config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
	config.dma_buf_count = 8;
	config.dma_buf_len = 512;
	config.use_apll = false;
	config.tx_desc_auto_clear = false;
	config.fixed_mclk = 0;
	i2s_pin_config_t pins = {};
	pins.bck_io_num = I2S_SCK_PIN;
	pins.ws_io_num = I2S_WS_PIN;
	pins.data_out_num = I2S_PIN_NO_CHANGE;
	pins.data_in_num = I2S_SD_PIN;
	i2s_driver_install(I2S_PORT, &config, 0, nullptr);
	i2s_set_pin(I2S_PORT, &pins);
}

static void read_audio_window() {
	size_t bytes_read = 0;
	size_t bytes_remaining = CLIP_SAMPLES * sizeof(audio_buffer[0]);
	uint8_t* destination = reinterpret_cast<uint8_t*>(audio_buffer);
	while (bytes_remaining > 0) {
		i2s_read(I2S_PORT, destination, bytes_remaining, &bytes_read, portMAX_DELAY);
		destination += bytes_read;
		bytes_remaining -= bytes_read;
	}
}

static void measure_audio_window(float& rms, int16_t& minimum, int16_t& maximum) {
	double sum_squared = 0.0;
	minimum = INT16_MAX;
	maximum = INT16_MIN;
	for (int i = 0; i < CLIP_SAMPLES; ++i) {
		int16_t sample = audio_buffer[i];
		if (sample < minimum) minimum = sample;
		if (sample > maximum) maximum = sample;
		sum_squared += static_cast<double>(sample) * sample;
	}
	rms = sqrtf(static_cast<float>(sum_squared / CLIP_SAMPLES));
}

static void print_feature_debug() {
	float minimum = mfcc[0];
	float maximum = mfcc[0];
	float mean = 0.0f;
	for (int i = 0; i < NUM_FRAMES * NUM_MFCC; ++i) {
		minimum = fminf(minimum, mfcc[i]);
		maximum = fmaxf(maximum, mfcc[i]);
		mean += mfcc[i];
	}
	mean /= NUM_FRAMES * NUM_MFCC;
	log_line("[MFCC] min=%.3f max=%.3f mean=%.3f first=[%.3f,%.3f,%.3f]",
			  minimum, maximum, mean, mfcc[0], mfcc[1], mfcc[2]);
}

static void print_layer_debug() {
	float minimum = tensor_a[0];
	float maximum = tensor_a[0];
	float mean = 0.0f;
	for (int i = 0; i < 48 * 13 * 13; ++i) {
		minimum = fminf(minimum, tensor_a[i]);
		maximum = fmaxf(maximum, tensor_a[i]);
		mean += tensor_a[i];
	}
	mean /= 48.0f * 13.0f * 13.0f;
	log_line("[MODEL] final feature min=%.5f max=%.5f mean=%.5f first=%.5f",
			  minimum, maximum, mean, tensor_a[0]);
}

void setup() {
	Serial.begin(115200);
	delay(500);
	audio_buffer = static_cast<int16_t*>(malloc(CLIP_SAMPLES * sizeof(int16_t)));
	mfcc = static_cast<float*>(malloc(NUM_FRAMES * NUM_MFCC * sizeof(float)));
	mel_filters = static_cast<float (*)[FFT_SIZE / 2 + 1]>(
		malloc(NUM_MELS * (FFT_SIZE / 2 + 1) * sizeof(float)));
	fft_real = static_cast<float*>(malloc(FFT_SIZE * sizeof(float)));
	fft_imag = static_cast<float*>(malloc(FFT_SIZE * sizeof(float)));
	power_spectrum = static_cast<float*>(malloc((FFT_SIZE / 2 + 1) * sizeof(float)));
	mel_energy = static_cast<float*>(malloc(NUM_MELS * sizeof(float)));
	tensor_a = static_cast<float*>(malloc(MAX_TENSOR_VALUES * sizeof(float)));
	tensor_b = static_cast<float*>(malloc(MAX_TENSOR_VALUES * sizeof(float)));
	if (!audio_buffer || !mfcc || !mel_filters || !fft_real || !fft_imag ||
		!power_spectrum || !mel_energy || !tensor_a || !tensor_b) {
		Serial.println("ERROR: insufficient heap for KWS buffers");
		while (true) delay(1000);
	}
	initialize_features();
	initialize_i2s();
	streamer.begin(WIFI_SSID, WIFI_PASSWORD, ASR_SERVER_HOST, ASR_SERVER_PORT, "/ws/audio");
	log_line("antriksya KWS ready: heap=%u bytes, I2S pins BCLK=%d WS=%d SD=%d",
			  ESP.getFreeHeap(), I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN);
}

void loop() {
	streamer.loop();  // must be called every iteration to keep the socket alive

	read_audio_window();
	float rms = 0.0f;
	int16_t minimum = 0;
	int16_t maximum = 0;
	measure_audio_window(rms, minimum, maximum);
	uint32_t feature_start = micros();
	extract_mfcc();
	uint32_t feature_time = micros() - feature_start;
	float probabilities[KWS_NUM_CLASSES];
	uint32_t model_start = micros();
	int prediction = run_model(probabilities);
	uint32_t model_time = micros() - model_start;
	log_line("[AUDIO] rms=%.1f norm=%.4f peak=[%d,%d] samples=[%d,%d,%d,%d]",
			  rms, rms / 32768.0f, minimum, maximum,
			  audio_buffer[0], audio_buffer[1], audio_buffer[2], audio_buffer[3]);
	print_feature_debug();
	print_layer_debug();
	log_line("[TIMING] mfcc=%lu ms model=%lu ms heap=%u",
			  feature_time / 1000, model_time / 1000, ESP.getFreeHeap());
	log_line("[PROB] antriksya: %.4f unknown: %.4f silence: %.4f -> %s",
			  probabilities[0], probabilities[1], probabilities[2],
			  CLASS_NAMES[prediction]);

	// Duty cycle of the capture window spent on feature extraction + inference.
	uint32_t compute_time_us = feature_time + model_time;
	float cpu_percent = 100.0f * compute_time_us / 1000000.0f;
	streamer.sendMetrics(CLASS_NAMES[prediction], cpu_percent);

	bool wake_detected = wake_threshold_crossed(
		probabilities[WAKE_CLASS_INDEX]);
	if (prediction == WAKE_CLASS_INDEX && wake_detected) {
		log_line("detected");
		streamer.wakeDetected(CLASS_NAMES[WAKE_CLASS_INDEX]);
		streamer.sendAudio(audio_buffer, CLIP_SAMPLES);
		streamer.endStream(CLIP_SAMPLES * 1000UL / SAMPLE_RATE);
	}
}
