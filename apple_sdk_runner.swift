// apple_sdk_runner.swift
// Usage: swift apple_sdk_runner.swift <input_json> <output_json>
// Requires: macOS 26+, Xcode 26+, Apple Intelligence enabled

import Foundation
import FoundationModels

let args = CommandLine.arguments
guard args.count == 3 else {
    fputs("Usage: swift apple_sdk_runner.swift <input.json> <output.json>\n", stderr)
    exit(1)
}

let inputURL  = URL(fileURLWithPath: args[1])
let outputURL = URL(fileURLWithPath: args[2])

// Load utterances from JSON
let inputData = try Data(contentsOf: inputURL)
let utterances = try JSONDecoder().decode([String].self, from: inputData)

let intents = [
    "alarm_query", "alarm_remove", "alarm_set",
    "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_query", "calendar_remove", "calendar_set",
    "cooking_query", "cooking_recipe",
    "datetime_convert", "datetime_query",
    "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
    "general_greet", "general_joke", "general_quirky",
    "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
    "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup",
    "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove",
    "music_dislikeness", "music_likeness", "music_query",
    "music_settings", "news_query",
    "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
    "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies",
    "social_post", "social_query",
    "takeaway_order", "takeaway_query",
    "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
    "weather_query"]

let intentList = intents.joined(separator: "\n")

@Generable
struct IntentPrediction {
    var intent: String
}

let model = SystemLanguageModel.default
guard case .available = model.availability else {
    fputs("Apple Intelligence not available on this device.\n", stderr)
    exit(1)
}

var predictions: [String] = []

for (i, utterance) in utterances.enumerated() {
    let session = LanguageModelSession {
        """
        You are an intent classifier. Given a user utterance, classify it into
        exactly one of the following intents:

        \(intentList)

        Reply with only the intent label, exactly as written above. No explanation.
        """
    }

    do {
        let result = try await session.respond(
            to: "Utterance: \(utterance)",
            generating: IntentPrediction.self
        )
        predictions.append(result.content.intent)
    } catch {
        predictions.append("unknown")
    }

    if (i + 1) % 50 == 0 {
        fputs("Progress: \(i + 1)/\(utterances.count)\n", stderr)
    }
}

let outputData = try JSONEncoder().encode(predictions)
try outputData.write(to: outputURL)
fputs("Done. Predictions written to \(args[2])\n", stderr)