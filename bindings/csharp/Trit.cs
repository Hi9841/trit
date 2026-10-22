// Trit for .NET: speech to text on the CPU, through the C ABI of trit-ffi.
//
//   using var engine = new Trit("whisper-large-v3-turbo");  // folder, `trit models` name, or null
//   Console.WriteLine(engine.Kind);                          // "whisper"
//   Console.WriteLine(engine.TranscribeWav("take.wav"));      // the model detects the language
//   Console.WriteLine(engine.Transcribe(samples, "de"));      // float[] at 16 kHz mono
//
//   using var cleaner = new TritCleaner();                   // Superwhisper S1-mini
//   Console.WriteLine(cleaner.Clean("so um send it friday no wait thursday"));
//
// Copy trit_ffi.dll (libtrit_ffi.so, libtrit_ffi.dylib) next to your program.
// Build it with `cargo build --release -p trit-ffi` in the trit folder.
// Add <AllowUnsafeBlocks>true</AllowUnsafeBlocks> to the project file.

using System;
using System.Runtime.InteropServices;

namespace TritSpeech;

public sealed class TritException(string message) : Exception(message);

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
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static unsafe partial IntPtr trit_transcribe_with(IntPtr engine, float* samples, nuint count, string? language, int beam);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_transcribe_wav(IntPtr engine, string path, string? language);
    [LibraryImport(Lib)] public static partial void trit_free_text(IntPtr text);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int trit_set_dictionary(IntPtr engine, string terms);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_cleaner_load_with(string? model, string? precision);
    [LibraryImport(Lib)] public static partial void trit_cleaner_free(IntPtr cleaner);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial IntPtr trit_clean(IntPtr cleaner, string text, string? styling, string? structure, string? context);
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
    /// for Whisper, Moonshine, and Canary (0: the model's default, 5 for Whisper and Moonshine; 1: greedy).</summary>
    public string Transcribe(ReadOnlySpan<float> samples, string? language = null, int beam = 0)
    {
        unsafe
        {
            fixed (float* pointer = samples)
                return Native.Text(Native.trit_transcribe_with(Handle, pointer, (nuint)samples.Length, language, beam));
        }
    }

    /// <summary>A 16 kHz WAV file (16-bit PCM or 32-bit float).</summary>
    public string TranscribeWav(string path, string? language = null) =>
        Native.Text(Native.trit_transcribe_wav(Handle, path, language));

    /// <summary>Terms to favor while decoding (Parakeet). Returns how many it cannot spell.</summary>
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
