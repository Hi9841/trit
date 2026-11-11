// Trit for .NET: speech to text on the CPU or the GPU and text to speech on the CPU, through the C ABI of trit-ffi.
//
//   using var engine = new Trit("whisper-large-v3-turbo");  // folder, `trit models` name, or null
//   Console.WriteLine(Trit.GpuInfo());                       // "NVIDIA GeForce RTX 4090, Dx12, f16"
//   Console.WriteLine(engine.Kind);                          // "whisper"
//   Console.WriteLine(engine.TranscribeWav("take.wav"));      // the model detects the language
//   Console.WriteLine(engine.Transcribe(samples, "de"));      // float[] at 16 kHz mono
//   Console.WriteLine(engine.Transcribe(mic, "en", sampleRate: 48000));  // another rate: Trit resamples
//   Console.WriteLine(engine.TranscribeWav("take.wav", task: "translate"));  // into English
//   foreach (var s in engine.SegmentsWav("take.wav")) Console.WriteLine($"{s.Start}-{s.End} {s.Text}");
//
//   using var cleaner = new TritCleaner();                   // Superwhisper S1-mini
//   Console.WriteLine(cleaner.Clean("so um send it friday no wait thursday"));
//
//   using var voice = new TritVoice();                       // Kokoro-82M, else MMS-TTS English
//   voice.SpeakWav("The build passed on March 3rd.", "speech.wav", voice: "bf_emma");
//   float[] samples = voice.Speak("Ship it.", speed: 1.1f);  // at voice.SampleRate
//   voice.SpeakStream(longText, part => { Play(part); return true; });  // each part as it is made
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
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_load_json(string? model, string? options);
    [LibraryImport(Lib)] public static partial int trit_cancel(IntPtr engine);
    [LibraryImport(Lib)] public static unsafe partial void trit_set_log(delegate* unmanaged[Cdecl]<byte*, IntPtr, void> callback, IntPtr userData);
    [LibraryImport(Lib)] public static partial IntPtr trit_gpu_info();
    [LibraryImport(Lib)] public static partial void trit_release_memory();
    [LibraryImport(Lib)] public static partial IntPtr trit_kind(IntPtr engine);
    [LibraryImport(Lib)] public static partial void trit_free(IntPtr engine);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial IntPtr trit_transcribe_json(IntPtr engine, float* samples, nuint count, string? options);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_transcribe_wav_json(IntPtr engine, string path, string? options);
    [LibraryImport(Lib)] public static partial void trit_free_text(IntPtr text);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int trit_set_dictionary(IntPtr engine, string terms);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_cleaner_load_ex(string? model, string? precision, string? device);
    [LibraryImport(Lib)] public static partial void trit_cleaner_free(IntPtr cleaner);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_clean(IntPtr cleaner, string text, string? styling, string? structure, string? context);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_voice_load(string? model);
    [LibraryImport(Lib)] public static partial void trit_voice_free(IntPtr voice);
    [LibraryImport(Lib)] public static partial int trit_voice_sample_rate(IntPtr voice);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial float* trit_speak(IntPtr voice, string text, string? options, out nuint count);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int trit_speak_wav(IntPtr voice, string text, string path, string? options);
    [LibraryImport(Lib)] public static unsafe partial void trit_free_samples(float* samples, nuint count);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial int trit_speak_stream(IntPtr voice, string text, string? options, delegate* unmanaged[Cdecl]<float*, nuint, IntPtr, int> callback, IntPtr userData);
    [LibraryImport(Lib)] public static partial IntPtr trit_voice_list(IntPtr voice);
    [LibraryImport(Lib)] public static partial IntPtr trit_voice_kind(IntPtr voice);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_phonemes(IntPtr voice, string text, string? options);
}

/// <summary>One loaded speech model. Load once and keep it.</summary>
public sealed class Trit : IDisposable
{
    IntPtr _handle;

    /// <summary>A model folder, a `trit models` name, or null for the default. precision: "auto" (null;
    /// int8 for layers 384 wide and more, f16 below; a GGUF file keeps its stored types), "int8" (least memory),
    /// "f16", "f32" (as the reference), or "q8_0" (ggml's Q8_0).
    /// device: "cpu" (default), "gpu", or "auto" (the GPU for the Parakeet models when the computer has an
    /// adapter; the CPU for the other families, which are slower on the GPU).
    /// TRIT_DEVICE overrides device. cache: keep the converted weights in the folder .trit of the model and
    /// map that file, so they are file memory and not private memory (the first load writes it; the model
    /// must be on a local disk).</summary>
    public Trit(string? model = null, string? precision = null, string device = "cpu", bool cache = false)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            if (precision != null) json.WriteString("precision", precision);
            json.WriteString("device", device);
            json.WriteBoolean("cache", cache);
            json.WriteEndObject();
        }
        _handle = Native.trit_load_json(model, System.Text.Encoding.UTF8.GetString(buffer.ToArray()));
        if (_handle == IntPtr.Zero) throw Native.Error();
    }

    /// <summary>Stops the calls that run on this engine now, on other threads: each throws TritException
    /// ("the call was cancelled"). Parakeet stops in a part of a second, the other families at the end of
    /// their window. Returns the number of calls it stopped.</summary>
    public int Cancel() => Native.trit_cancel(Handle);

    static Action<string>? _log;

    [UnmanagedCallersOnly(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    static unsafe void OnLog(byte* message, IntPtr userData)
    {
        // An exception must not cross the native frames.
        try { _log?.Invoke(Marshal.PtrToStringUTF8((IntPtr)message) ?? ""); }
        catch (Exception) { }
    }

    /// <summary>Sends the engine's messages (a GPU failure and the change to the CPU) to log; null sends them
    /// to stderr again. The function can run on any thread, and must not call Trit.</summary>
    public static void SetLog(Action<string>? log)
    {
        _log = log;
        unsafe { Native.trit_set_log(log == null ? null : &OnLog, IntPtr.Zero); }
    }

    /// <summary>The GPU adapter, for example "NVIDIA GeForce RTX 4090, Dx12, f16".
    /// Throws when the computer has no adapter.</summary>
    public static string GpuInfo() => Native.Text(Native.trit_gpu_info());

    public static string Version => Marshal.PtrToStringUTF8(Native.trit_version())!;

    /// <summary>Returns memory freed by earlier calls to the system; call it when the app goes idle.</summary>
    public static void ReleaseMemory() => Native.trit_release_memory();

    /// <summary>The `model_type` of the model, for example "parakeet_tdt".</summary>
    public string Kind => Marshal.PtrToStringUTF8(Native.trit_kind(Handle))!;

    /// <summary>Mono samples in -1..1; language is an ISO 639-1 code or null to detect it; beam: beams
    /// for Whisper, Moonshine, and Canary (0: the model's default, 5 for Whisper and Moonshine; 1: greedy);
    /// task: "transcribe" (null), or "translate" into English (multilingual Whisper, Canary); fallback:
    /// Whisper decodes a window again at higher temperatures when its text repeats itself (null: on);
    /// sampleRate: the rate of the samples in Hz (0: 16000). Trit resamples another rate, such as the 44100
    /// or 48000 of a microphone. Samples at another rate with no sampleRate give wrong text.</summary>
    public string Transcribe(ReadOnlySpan<float> samples, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null, int sampleRate = 0) =>
        Text(Run(samples, Options(language, beam, task, fallback, false, sampleRate)));

    /// <summary>A WAV file (16, 24, or 32-bit PCM, or 32-bit float; a file that is not at 16 kHz is
    /// resampled); the rest as for <see cref="Transcribe"/>.</summary>
    public string TranscribeWav(string path, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        Text(RunWav(path, Options(language, beam, task, fallback, false)));

    /// <summary>The text in timed segments: sentences or pauses for Parakeet, windows of up to 30 s (40 s
    /// Canary) for the others. Arguments as for <see cref="Transcribe"/>.</summary>
    public List<Segment> Segments(ReadOnlySpan<float> samples, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null, int sampleRate = 0) =>
        SegmentList(Run(samples, Options(language, beam, task, fallback, true, sampleRate)));

    /// <summary><see cref="Segments"/> for a WAV file.</summary>
    public List<Segment> SegmentsWav(string path, string? language = null, int beam = 0,
        string? task = null, bool? fallback = null) =>
        SegmentList(RunWav(path, Options(language, beam, task, fallback, true)));

    /// <summary>The options JSON of trit_transcribe_json.</summary>
    static string Options(string? language, int beam, string? task, bool? fallback, bool timestamps,
        int sampleRate = 0)
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
            // An older library does not know the key.
            if (sampleRate > 0) json.WriteNumber("sample_rate", sampleRate);
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

    /// <summary>Frees the model. Do not call it while another thread is in a call on this engine.</summary>
    public void Dispose()
    {
        Native.trit_free(_handle);
        _handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    // A program that does not call Dispose must not keep the model for its whole life.
    ~Trit() => Native.trit_free(_handle);

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(Trit));
}

/// <summary>Superwhisper S1-mini: a raw transcript to clean written English.</summary>
public sealed class TritCleaner : IDisposable
{
    IntPtr _handle;

    /// <summary>precision and device as for <see cref="Trit"/>.</summary>
    public TritCleaner(string? model = null, string? precision = null, string device = "cpu")
    {
        _handle = Native.trit_cleaner_load_ex(model, precision, device);
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
        GC.SuppressFinalize(this);
    }

    ~TritCleaner() => Native.trit_cleaner_free(_handle);

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(TritCleaner));
}

/// <summary>A voice: Kokoro-82M (24 kHz, 28 English voices; reads numbers and symbols) or a VITS voice
/// (Meta MMS-TTS, 16 kHz; reads letters only).</summary>
public sealed class TritVoice : IDisposable
{
    IntPtr _handle;

    /// <summary>A voice folder, a `trit models` name ("kokoro-82m", "mms-tts-eng"), or null for the default
    /// (Kokoro, then MMS-TTS).</summary>
    public TritVoice(string? model = null)
    {
        _handle = Native.trit_voice_load(model);
        if (_handle == IntPtr.Zero) throw Native.Error();
    }

    /// <summary>The rate of the samples of <see cref="Speak"/> (24000 Kokoro, 16000 MMS-TTS).</summary>
    public int SampleRate => Native.trit_voice_sample_rate(Handle);

    /// <summary>"kokoro" or "vits".</summary>
    public string Kind => Marshal.PtrToStringUTF8(Native.trit_voice_kind(Handle))!;

    /// <summary>The voices of a Kokoro folder: af_/am_ American, bf_/bm_ British (none for VITS).</summary>
    public string[] Voices
    {
        get
        {
            var text = Native.Text(Native.trit_voice_list(Handle));
            return text.Length == 0 ? [] : text.Split('\n');
        }
    }

    /// <summary>The phonemes Kokoro speaks for text, one line per pass, in the accent of voice.</summary>
    public string Phonemes(string text, string? voice = null) =>
        Native.Text(Native.trit_phonemes(Handle, text, Options(voice, null, null, null, 0, false)));

    /// <summary>Mono samples in -1..1 at <see cref="SampleRate"/>. voice: Kokoro voice (null: af_heart; "a,b"
    /// blends two); speed: 1 is the voice's pace, 1.25 faster, from 0.25 to 4 (null: the voice's own); noise: how much the
    /// voice varies; noiseDuration: VITS timing; one text and one seed give the same audio; phonemes: Kokoro,
    /// the text is phonemes.</summary>
    public float[] Speak(string text, string? voice = null, float? speed = null, float? noise = null,
        float? noiseDuration = null, ulong seed = 0, bool phonemes = false)
    {
        unsafe
        {
            float* samples = Native.trit_speak(Handle, text, Options(voice, speed, noise, noiseDuration, seed, phonemes), out nuint count);
            if (samples == null) throw Native.Error();
            try { return new ReadOnlySpan<float>(samples, checked((int)count)).ToArray(); }
            finally { Native.trit_free_samples(samples, count); }
        }
    }

    /// <summary><see cref="Speak"/> into a 16-bit mono WAV file.</summary>
    public void SpeakWav(string text, string path, string? voice = null, float? speed = null, float? noise = null,
        float? noiseDuration = null, ulong seed = 0, bool phonemes = false)
    {
        if (Native.trit_speak_wav(Handle, text, path, Options(voice, speed, noise, noiseDuration, seed, phonemes)) < 0)
            throw Native.Error();
    }

    /// <summary>One SpeakStream call: the caller's function, and the exception it threw.</summary>
    sealed class StreamCall(Func<float[], bool> onPart)
    {
        public readonly Func<float[], bool> OnPart = onPart;
        public Exception? Error;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    static unsafe int OnPart(float* samples, nuint count, IntPtr userData)
    {
        var call = (StreamCall)GCHandle.FromIntPtr(userData).Target!;
        // An exception must not cross the native frames: .NET then stops the
        // process. Keep it, stop the stream, and throw it after the call.
        try
        {
            return call.OnPart(new ReadOnlySpan<float>(samples, checked((int)count)).ToArray()) ? 1 : 0;
        }
        catch (Exception error)
        {
            call.Error = error;
            return 0;
        }
    }

    /// <summary><see cref="Speak"/>, part by part: onPart gets each part (a sentence for VITS, a line or a
    /// 510-phoneme pass for Kokoro) as soon as it is made; return false to stop.</summary>
    public void SpeakStream(string text, Func<float[], bool> onPart, string? voice = null, float? speed = null,
        float? noise = null, float? noiseDuration = null, ulong seed = 0, bool phonemes = false)
    {
        var call = new StreamCall(onPart);
        var handle = GCHandle.Alloc(call);
        try
        {
            int status;
            unsafe
            {
                status = Native.trit_speak_stream(Handle, text, Options(voice, speed, noise, noiseDuration, seed, phonemes), &OnPart, GCHandle.ToIntPtr(handle));
            }
            if (call.Error != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(call.Error).Throw();
            if (status < 0) throw Native.Error();
        }
        finally { handle.Free(); }
    }

    /// <summary>The options JSON of trit_speak.</summary>
    static string Options(string? voice, float? speed, float? noise, float? noiseDuration, ulong seed, bool phonemes)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            if (voice != null) json.WriteString("voice", voice);
            if (speed is float s) json.WriteNumber("speed", s);
            if (noise is float n) json.WriteNumber("noise", n);
            if (noiseDuration is float d) json.WriteNumber("noise_duration", d);
            json.WriteNumber("seed", seed);
            json.WriteBoolean("phonemes", phonemes);
            json.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    public void Dispose()
    {
        Native.trit_voice_free(_handle);
        _handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    ~TritVoice() => Native.trit_voice_free(_handle);

    IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(TritVoice));
}
