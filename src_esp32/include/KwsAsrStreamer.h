#pragma once

#include <Arduino.h>
#include <WebSocketsClient.h>

class KwsAsrStreamer {
public:
	void begin(const char* ssid, const char* password, const char* host,
			  uint16_t port, const char* path);
	void loop();
	void sendLog(const char* text);
	void sendMetrics(const char* state, float cpu_percent);
	void wakeDetected(const char* keyword);
	void sendAudio(const int16_t* samples, size_t sample_count);
	void endStream(uint32_t duration_ms);

private:
	WebSocketsClient websocket;
	bool connected = false;
	String wifi_ssid;
	String wifi_password;
	String server_host;
	String server_path;
	uint16_t server_port = 0;
	unsigned long last_diagnostic = 0;
	unsigned long last_wifi_retry = 0;
	unsigned long last_scan = 0;

	void sendEvent(const String& event);
	void printNetworkDiagnostic();
	static String escapeJson(const char* text);
	static void handleEvent(WStype_t type, uint8_t* payload, size_t length);
};
