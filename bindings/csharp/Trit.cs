// Trit for .NET: speech to text and text to speech on the CPU, through the C ABI of trit-ffi.
//
//   using var engine = new Trit("whisper-large-v3-turbo");  // folder, `trit models` name, or null
//   Console.WriteLine(engine.Kind);                          // "whisper"
//   Console.WriteLine(engine.TranscribeWav("take.wav"));      // the model detects the language
//   Console.WriteLine(engine.Transcribe(samples, "de"));      // float[] at 16 kHz mono
//   Console.WriteLine(engine.TranscribeWav("take.wav", task: "translate"));  // into English
//   foreach (var s in engine.SegmentsWav("take.wav")) Console.WriteLine($"{s.Start}-{s.End} {s.Text}");
//
//   using var cleaner = new TritCleaner();                   // Superwhisper S1-mini
//   Console.WriteLine(cleaner.Clean("so um send it friday no wait thursday"));
//
//   using var voice = new TritVoice();                       // Meta MMS-TTS English
//   voice.SpeakWav("The build passed.", "speech.wav");
//   float[] samples = voice.Speak("Ship it.", speed: 1.1f);  // at voice.SampleRate
//
// Copy trit_ffi.dll (libtrit_ffi.so, libtrit_ffi.dylib) next to your program.
// Build it with `cargo build --release -p trit-ffi` in the trit folder.
// Add <AllowUnsafeBlocks>true</AllowUnsafeBlocks> to the project file.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace TritSpeech;

public sealed class TritException(string message) : Exception(message);

/// <summary>A stretch of the recording and its text; times in seconds from the start.</summary>
public sealed record Segment(double Start, double End, string Text);

internal static partial class Native
{
    const string Lib = "trit_ffi";

    public static TritException Error() => new(Marshal.PtrToStringUTF8(trit_last_error()) ?? "unknown error");

    public static string Text(IntPtr text)
    {
        if (text == IntPtr.Zero) throw Error();
        try { return Marshal.PtrToStringUTF8(text)!; }
        finally { trit_free_text(text); }
    }

    [LibraryImport(Lib)] public static partial IntPtr trit_version();
    [LibraryImport(Lib)] public static partial IntPtr trit_last_error();
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_load_with(string? model, string? precision);
    [LibraryImport(Lib)] public static partial void trit_release_memory();
    [LibraryImport(Lib)] public static partial IntPtr trit_kind(IntPtr engine);
    [LibraryImport(Lib)] public static partial void trit_free(IntPtr engine);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial IntPtr trit_transcribe_json(IntPtr engine, float* samples, nuint count, string? options);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_transcribe_wav_json(IntPtr engine, string path, string? options);
    [LibraryImport(Lib)] public static partial void trit_free_text(IntPtr text);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int trit_set_dictionary(IntPtr engine, string terms);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_cleaner_load_with(string? model, string? precision);
    [LibraryImport(Lib)] public static partial void trit_cleaner_free(IntPtr cleaner);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_clean(IntPtr cleaner, string text, string? styling, string? structure, string? context);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_voice_load(string? model);
    [LibraryImport(Lib)] public static partial void trit_voice_free(IntPtr voice);
    [LibraryImport(Lib)] public static partial int trit_voice_sample_rate(IntPtr voice);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial float* trit_speak(IntPtr voice, string text, string? options, out nuint count);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int trit_speak_wav(IntPtr voice, string text, string path, string? options);
    [LibraryImport(Lib)] public static unsafe partial void trit_free_samples(float* samples, nuint count);
}

/// <summary>One loaded speech model. Load once and keep it.</summary>
public sealed class Trit : IDisposable
{
    IntPtr _handle;

    /// <summary>A model folder, a `trit models` name, or null for the default. precision: "auto" (null;
    /// int8 for layers 384 wide and more, f16 below), "int8" (least memory), "f16", or "f32" (as the reference).</summary>
    public Trit(string? model = null, string? precision = null)
    {
        _handle = Native.trit_load_with(model, precision);
        if (_handle == IntPtr.Zero) throw Native.Error();
    }

    public static string Version => Marshal.PtrToStringUTF8(Native.trit_version())!;

    /// <summary>Returns memory freed by earlier calls to the system; call it when the app goes idle.</summary>
    public static void ReleaseMemory() => Native.trit_release_memory();

    /// <summary>"parakeet_tdt", "parakeet_ctc", "canary", "whisper", or "moonshine".</summary>
    public string Kind => Marshal.PtrToStringUTF8(Native.trit_kind(Handle))!;

    /// <summary>16 kHz mono samples in -1..1; language is an ISO 639-1 code or null to detect it; beam: beams
    /// for Whisper, Moonshine, and Canary (0: the model's default, 5 for Whisper and Moonshine; 1: greedy);
    /// task: "transcribe" (null), or "translate" into English (multilingual Whisper, Canary); fallback:
    /// Whisper decodes a window again at higher temperatures when its text repeats itself (null: on).</summary>
    public string Transcribe(ReadOnlySpan<float> samples, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        Text(Run(samples, Options(language, beam, task, fallback, false)));

    /// <summary>A 16 kHz WAV file (16-bit PCM or 32-bit float); the rest as for <see cref="Transcribe"/>.</summary>
    public string TranscribeWav(string path, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        Text(RunWav(path, Options(language, beam, task, fallback, false)));

    /// <summary>The text in timed segments: sentences or pauses for Parakeet, windows of up to 30 s (40 s
    /// Canary) for the others. Arguments as for <see cref="Transcribe"/>.</summary>
    public List<Segment> Segments(ReadOnlySpan<float> samples, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        SegmentList(Run(samples, Options(language, beam, task, fallback, true)));

    /// <summary><see cref="Segments"/> for a 16 kHz WAV file.</summary>
    public List<Segment> SegmentsWav(string path, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        SegmentList(RunWav(path, Options(language, beam, task, fallback, true)));

    /// <summary>The options JSON of trit_transcribe_json.</summary>
    static string Options(string? language, int beam, string? task, bool? fallback, bool timestamps)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            if (language != null) json.WriteString("language", language);
            json.WriteNumber("beam", beam);
            if (task != null) json.WriteString("task", task);
            if (fallback is bool on) json.WriteBoolean("fallback", on);
            json.WriteBoolean("timestamps", timestamps);
            json.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    JsonDocument Run(ReadOnlySpan<float> samples, string options)
    {
        unsafe
        {
            fixed (float* pointer = samples)
                return JsonDocument.Parse(Native.Text(Native.trit_transcribe_json(Handle, pointer, (nuint)samples.Length, options)));
        }
    }

    JsonDocument RunWav(string path, string options) =>
        JsonDocument.Parse(Native.Text(Native.trit_transcribe_wav_json(Handle, path, options)));

    static string Text(JsonDocument result)
    {
        using (result) return result.RootElement.GetProperty("text").GetString()!;
    }

    static List<Segment> SegmentList(JsonDocument result)
    {
        using (result)
        {
            var segments = new List<Segment>();
            foreach (var s in result.RootElement.GetProperty("segments").EnumerateArray())
                segments.Add(new Segment(s.GetProperty("start").GetDouble(), s.GetProperty("end").GetDouble(),
                    s.GetProperty("text").GetString()!));
            return segments;
        }
    }

    /// <summary>Terms to favor while decoding (Parakeet). Returns how many it cannot spell. No terms clears them.</summary>
    public int SetDictionary(params string[] terms)
    {
        int skipped = Native.trit_set_dictionary(Handle, string.Join('\n', terms));
        return skipped < 0 ? throw Native.Error() : skipped;
    }

    public void Dispose()
    {
        Native.trit_free(_handle);
        _handle = IntPtr.Zero;
    }

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(Trit));
}

/// <summary>Superwhisper S1-mini: a raw transcript to clean written English.</summary>
public sealed class TritCleaner : IDisposable
{
    IntPtr _handle;

    /// <summary>precision as for <see cref="Trit"/>.</summary>
    public TritCleaner(string? model = null, string? precision = null)
    {
        _handle = Native.trit_cleaner_load_with(model, precision);
        if (_handle == IntPtr.Zero) throw Native.Error();
    }

    /// <summary>styling: casual, semi-casual, semi-formal (default), formal; structure: prose (default),
    /// lists; context: general (default), email.</summary>
    public string Clean(string text, string? styling = null, string? structure = null, string? context = null) =>
        Native.Text(Native.trit_clean(Handle, text, styling, structure, context));

    public void Dispose()
    {
        Native.trit_cleaner_free(_handle);
        _handle = IntPtr.Zero;
    }

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(TritCleaner));
}

/// <summary>A VITS voice (Meta MMS-TTS): text to speech. It reads letters; write numbers and symbols as words.</summary>
public sealed class TritVoice : IDisposable
{
    IntPtr _handle;

    /// <summary>A voice folder, a `trit models` name ("mms-tts-eng"), or null for the default.</summary>
    public TritVoice(string? model = null)
    {
        _handle = Native.trit_voice_load(model);
        if (_handle == IntPtr.Zero) throw Native.Error();
    }

    /// <summary>The rate of the samples of <see cref="Speak"/> (16000 for MMS-TTS).</summary>
    public int SampleRate => Native.trit_voice_sample_rate(Handle);

    /// <summary>Mono samples in -1..1 at <see cref="SampleRate"/>. speed: 1 is the voice's pace, 1.25 faster
    /// (null: the voice's own); noise, noiseDuration: how much the voice and the timing vary (null: the
    /// voice's own); one text and one seed give the same audio.</summary>
    public float[] Speak(string text, float? speed = null, float? noise = null, float? noiseDuration = null,
        ulong seed = 0)
    {
        unsafe
        {
            float* samples = Native.trit_speak(Handle, text, Options(speed, noise, noiseDuration, seed), out nuint count);
            if (samples == null) throw Native.Error();
            try { return new ReadOnlySpan<float>(samples, checked((int)count)).ToArray(); }
            finally { Native.trit_free_samples(samples, count); }
        }
    }

    /// <summary><see cref="Speak"/> into a 16-bit mono WAV file.</summary>
    public void SpeakWav(string text, string path, float? speed = null, float? noise = null,
        float? noiseDuration = null, ulong seed = 0)
    {
        if (Native.trit_speak_wav(Handle, text, path, Options(speed, noise, noiseDuration, seed)) < 0)
            throw Native.Error();
    }

    /// <summary>The options JSON of trit_speak.</summary>
    static string Options(float? speed, float? noise, float? noiseDuration, ulong seed)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            if (speed is float s) json.WriteNumber("speed", s);
            if (noise is float n) json.WriteNumber("noise", n);
            if (noiseDuration is float d) json.WriteNumber("noise_duration", d);
            json.WriteNumber("seed", seed);
            json.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    public void Dispose()
    {
        Native.trit_voice_free(_handle);
        _handle = IntPtr.Zero;
    }

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(TritVoice));
}
