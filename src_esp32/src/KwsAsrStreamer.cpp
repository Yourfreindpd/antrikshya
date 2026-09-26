#include "KwsAsrStreamer.h"

#include <WiFi.h>

namespace {
	KwsAsrStreamer* active_streamer = nullptr;
}

String KwsAsrStreamer::escapeJson(const char* text) {
	String escaped;
	if (!text) return escaped;
	for (const char* current = text; *current; ++current) {
		if (*current == '\\' || *current == '"') escaped += '\\';
		if (*current == '\n') {
			escaped += "\\n";
		} else if (*current == '\r') {
			escaped += "\\r";
		} else {
			escaped += *current;
		}
	}
	return escaped;
}

void KwsAsrStreamer::begin(const char* ssid, const char* password,
						   const char* host, uint16_t port, const char* path) {
	active_streamer = this;
	wifi_ssid = ssid;
	wifi_password = password;
	server_host = host;
	server_port = port;
	server_path = path;
	WiFi.mode(WIFI_STA);
	WiFi.setAutoReconnect(true);
	WiFi.begin(ssid, password);
	unsigned long started = millis();
	while (WiFi.status() != WL_CONNECTED && millis() - started < 15000) {
		delay(250);
	}
	websocket.begin(host, port, path, "arduino");
	websocket.onEvent(handleEvent);
	websocket.setReconnectInterval(5000);
	websocket.enableHeartbeat(15000, 3000, 2);
	Serial.printf("[NETWORK] WiFi=%s status=%d IP=%s gateway=%s RSSI=%d dBm server=%s:%u%s\n",
				  WiFi.status() == WL_CONNECTED ? "connected" : "failed",
				  static_cast<int>(WiFi.status()),
				  WiFi.localIP().toString().c_str(),
				  WiFi.gatewayIP().toString().c_str(), WiFi.RSSI(),
				  host, port, path);
}

void KwsAsrStreamer::loop() {
	bool wifi_connected = WiFi.status() == WL_CONNECTED;
	if (!wifi_connected && millis() - last_wifi_retry >= 5000) {
		last_wifi_retry = millis();
		Serial.printf("[NETWORK] Reconnecting WiFi SSID=%s status=%d\n",
					  wifi_ssid.c_str(), static_cast<int>(WiFi.status()));
		WiFi.begin(wifi_ssid.c_str(), wifi_password.c_str());
	}
	if (wifi_connected) websocket.loop();
	if (millis() - last_diagnostic >= 10000) {
		last_diagnostic = millis();
		printNetworkDiagnostic();
	}
}

void KwsAsrStreamer::printNetworkDiagnostic() {
	if (WiFi.status() != WL_CONNECTED) {
		Serial.printf(
			"[NETWORK][DEBUG] WiFi unavailable status=%d "
			"(1=NO_SSID, 4=CONNECT_FAILED, 6=DISCONNECTED) configured_ssid=%s "
			"visible_ssid=%s\n",
			static_cast<int>(WiFi.status()), wifi_ssid.c_str(), WiFi.SSID().c_str());
		if (millis() - last_scan >= 30000) {
			last_scan = millis();
			int count = WiFi.scanNetworks(false, true);
			Serial.printf("[NETWORK][DEBUG] visible_networks=%d\n", count);
			for (int index = 0; index < count; ++index) {
				Serial.printf("[NETWORK][DEBUG] network[%d] ssid=%s rssi=%d "
						  "channel=%d\n", index, WiFi.SSID(index).c_str(),
						  WiFi.RSSI(index), WiFi.channel(index));
			}
			WiFi.scanDelete();
		}
		return;
	}
	WiFiClient probe;
	unsigned long started = millis();
	bool reachable = probe.connect(server_host.c_str(), server_port);
	unsigned long elapsed = millis() - started;
	if (reachable) probe.stop();
	Serial.printf(
		"[NETWORK][DEBUG] ws=%s wifi_status=%d ip=%s gateway=%s RSSI=%d "
		"server=%s:%u reachable=%s probe_ms=%lu heap=%u\n",
		connected ? "connected" : "disconnected",
		static_cast<int>(WiFi.status()), WiFi.localIP().toString().c_str(),
		WiFi.gatewayIP().toString().c_str(), WiFi.RSSI(), server_host.c_str(),
		server_port, reachable ? "yes" : "no", elapsed, ESP.getFreeHeap());
}

void KwsAsrStreamer::sendEvent(const String& event) {
	if (connected) {
		String message = event;
		websocket.sendTXT(message);
	}
}

void KwsAsrStreamer::sendLog(const char* text) {
	String event = "{\"event\":\"esp32_log\",\"text\":\"";
	event += escapeJson(text);
	event += "\"}";
	sendEvent(event);
}

void KwsAsrStreamer::sendMetrics(const char* state, float cpu_percent) {
	String event = "{\"event\":\"device_metrics\",\"state\":\"";
	event += escapeJson(state);
	event += "\",\"cpu_percent\":";
	event += String(cpu_percent, 2);
	event += ",\"free_heap\":";
	event += String(ESP.getFreeHeap());
	event += ",\"heap_size\":";
	event += String(ESP.getHeapSize());
	event += "}";
	sendEvent(event);
}

void KwsAsrStreamer::wakeDetected(const char* keyword) {
	String event = "{\"event\":\"wake_detected\",\"keyword\":\"";
	event += escapeJson(keyword);
	event += "\"}";
	sendEvent(event);
}

void KwsAsrStreamer::sendAudio(const int16_t* samples, size_t sample_count) {
	if (connected && samples && sample_count > 0) {
		websocket.sendBIN(reinterpret_cast<const uint8_t*>(samples),
						  sample_count * sizeof(int16_t));
	}
}

void KwsAsrStreamer::endStream(uint32_t duration_ms) {
	String event = "{\"event\":\"stream_end\",\"duration_ms\":";
	event += String(duration_ms);
	event += "}";
	sendEvent(event);
}

void KwsAsrStreamer::handleEvent(WStype_t type, uint8_t* payload,
								 size_t length) {
	if (!active_streamer) return;
	if (type == WStype_CONNECTED) {
		active_streamer->connected = true;
		Serial.printf("[NETWORK] WebSocket connected to %s:%u%s\n",
					  active_streamer->server_host.c_str(),
					  active_streamer->server_port,
					  active_streamer->server_path.c_str());
	} else if (type == WStype_DISCONNECTED) {
		active_streamer->connected = false;
		Serial.println("[NETWORK] WebSocket disconnected");
		active_streamer->printNetworkDiagnostic();
	} else if (type == WStype_ERROR) {
		Serial.printf("[NETWORK][ERROR] WebSocket error payload=%.*s\n",
					  static_cast<int>(length), payload ? payload :
					  reinterpret_cast<uint8_t*>(const_cast<char*>("")));
	} else if (type == WStype_TEXT) {
		Serial.printf("[SERVER] %.*s\n", static_cast<int>(length), payload);
	}
}
