/* Transcribes the WAV files on the command line with the C ABI.
 * Build (from trit/, after `cargo build --release -p trit-ffi`):
 *   Windows: clang examples/transcribe.c -Iinclude target/release/trit_ffi.dll.lib -o target/release/transcribe.exe
 *   Linux:   cc examples/transcribe.c -Iinclude -Ltarget/release -ltrit_ffi -o transcribe
 */
#include <stdio.h>
#include "trit.h"

int main(int argc, char **argv) {
    Trit *engine = trit_load(NULL);
    if (!engine) {
        printf("error: %s\n", trit_last_error());
        return 1;
    }
    int failed = 0;
    for (int i = 1; i < argc; i++) {
        char *text = trit_transcribe_wav(engine, argv[i], NULL);
        if (!text) {
            printf("error: %s\n", trit_last_error());
            failed = 1;
            continue;
        }
        printf("%s: %s\n", argv[i], text);
        trit_free_text(text);
    }
    trit_free(engine);
    return failed;
}
