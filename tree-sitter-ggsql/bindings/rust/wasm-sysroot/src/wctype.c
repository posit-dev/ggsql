#include <wctype.h>

__attribute__((weak))
int iswlower(wint_t wch) {
    return (unsigned)wch - L'a' < 26;
}

__attribute__((weak))
int iswupper(wint_t wch) {
    return (unsigned)wch - L'A' < 26;
}

__attribute__((weak))
int iswpunct(wint_t wch) {
  return (wch >= 33 && wch <= 47) ||
         (wch >= 58 && wch <= 64) ||
         (wch >= 91 && wch <= 96) ||
         (wch >= 123 && wch <= 126);
}
