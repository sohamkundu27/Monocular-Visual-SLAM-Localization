// Python-compatible text formatting.
//
// Every artefact the pipeline writes (metrics.json, diagnostics.csv,
// config.resolved.yaml, run.log, the console summary) is text produced by
// Python's formatting rules. Reproducing those rules exactly is what makes
// the C++ outputs byte-comparable with the Python ones:
//
// * ``repr(float)``: shortest round-tripping digits, fixed notation for
//   decimal exponents in [-4, 16), scientific otherwise, always with a ".0"
//   or exponent so the text reads back as a float.
// * ``round(x, n)``: correctly rounded to n decimal places (round-half-even on
//   the exact binary value), returned as the nearest double.
// * ``format(x, spec)`` for the f/e/g/d specs this project uses.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mvslam::py {

/// ``repr(x)`` / ``str(x)`` for a float.
std::string repr(double x);

/// ``str(True)`` -> "True".
std::string repr(bool value);

/// ``repr(s)`` for a str: quoted and escaped the way Python prints it.
std::string repr_str(const std::string& s);

/// ``round(x, ndigits)``.
double round(double x, int ndigits);

/// ``format(x, spec)`` for specs of the form ``[width][,][.precision](f|e|g)``
/// (and an empty spec, which is ``str(x)``).
std::string format(double x, const std::string& spec);

/// ``format(n, spec)`` for ``d``, ``,``, ``,d`` and width variants.
std::string format(std::int64_t n, const std::string& spec);

/// printf-style formatting with Python's ``%`` semantics for the conversions
/// used here; NaN and infinities print as ``nan``/``inf`` without a sign
/// artefact, as in Python.
std::string sprintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

/// ``str.ljust`` / ``str.rjust``.
std::string ljust(const std::string& s, std::size_t width);
std::string rjust(const std::string& s, std::size_t width);

/// One row as ``csv.writer`` (excel dialect) would write it, including the
/// ``\r\n`` line terminator.
std::string csv_row(const std::vector<std::string>& fields);

/// ``np.savetxt`` row with ``fmt="%.12e"`` (or another single printf spec):
/// values joined by a space, terminated by ``\n``.
std::string savetxt_row(const double* values, std::size_t n, const char* fmt = "%.12e");

}  // namespace mvslam::py
