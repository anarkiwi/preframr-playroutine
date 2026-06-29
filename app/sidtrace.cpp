/*
 * sidtrace - byte-exact, cycle-stamped SID register + IRQ/NMI tracer.
 *
 * Part of preframr-playroutine. Built on libsidplayfp (GPL-2.0-or-later);
 * this program is therefore distributed under the GNU General Public License
 * version 2 or later.
 *
 * It plays a .sid tune exactly as sidplayfp would (same cycle-accurate C64
 * emulation) and emits two artifacts:
 *   <prefix>.bin   flat array of fixed 16-byte little-endian event records
 *                  (numpy: np.fromfile(path, dtype=preframr_playroutine.EVENT_DTYPE))
 *   <prefix>.json  metadata sidecar describing the tune, timing and schema.
 *
 * The .bin stream is the oracle: every value written to a SID chip, every
 * CIA/VIC interrupt-line assertion, and every CPU interrupt vector-through,
 * each stamped with the absolute event-scheduler cycle.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sidplayfp/sidplayfp.h>
#include <sidplayfp/SidTune.h>
#include <sidplayfp/SidConfig.h>
#include <sidplayfp/SidInfo.h>
#include <sidplayfp/SidTuneInfo.h>
#include <sidplayfp/builders/residfp.h>

#include "instrument.h"

namespace {

// Event record, 16 bytes, fields little-endian. Must match EVENT_DTYPE in
// the Python package.
enum EventType : uint8_t {
    EV_SID_WRITE  = 0,  // chip=sid index, reg=0..0x1f, value, addr=full SID addr
    EV_CIA_IRQ    = 1,  // chip=1 CIA1(IRQ)/2 CIA2(NMI), addr=timerA latch, aux=timerB latch
    EV_VIC_IRQ    = 2,  // addr=raster compare line, aux=current raster line
    EV_CPU_VECTOR = 3,  // value=vector kind (0xfe IRQ,0xfa NMI,0xfc RST), addr=handler PC
};

#pragma pack(push, 1)
struct Record {
    uint64_t cycle;
    uint8_t  etype;
    uint8_t  chip;
    uint8_t  reg;
    uint8_t  value;
    uint16_t addr;
    uint16_t aux;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 16, "Record must be 16 bytes");

class TraceSink : public libsidplayfp::InstrumentSink {
public:
    explicit TraceSink(FILE *f) : m_file(f) { m_buf.reserve(BUF_RECS); }

    void sidWrite(int64_t cycle, int chip, uint16_t base,
                  uint8_t reg, uint8_t value) override {
        push({static_cast<uint64_t>(cycle), EV_SID_WRITE,
              static_cast<uint8_t>(chip), reg, value,
              static_cast<uint16_t>(base + reg), 0});
    }

    void ciaIrq(int64_t cycle, int source, uint16_t taLatch, uint16_t tbLatch) override {
        push({static_cast<uint64_t>(cycle), EV_CIA_IRQ,
              static_cast<uint8_t>(source), 0, 0, taLatch, tbLatch});
    }

    void vicIrq(int64_t cycle, unsigned rasterY, unsigned rasterCmp) override {
        push({static_cast<uint64_t>(cycle), EV_VIC_IRQ, 3, 0, 0,
              static_cast<uint16_t>(rasterCmp), static_cast<uint16_t>(rasterY)});
    }

    void cpuVector(int64_t cycle, uint8_t kind, uint16_t pc) override {
        push({static_cast<uint64_t>(cycle), EV_CPU_VECTOR, 0, 0, kind, pc, 0});
    }

    void flush() {
        if (!m_buf.empty()) {
            std::fwrite(m_buf.data(), sizeof(Record), m_buf.size(), m_file);
            m_buf.clear();
        }
    }

    uint64_t count() const { return m_count; }

private:
    static const size_t BUF_RECS = 65536;

    void push(const Record &r) {
        m_buf.push_back(r);
        ++m_count;
        if (m_buf.size() >= BUF_RECS)
            flush();
    }

    FILE *m_file;
    std::vector<Record> m_buf;
    uint64_t m_count = 0;
};

struct Options {
    std::string input;
    std::string prefix;
    unsigned song = 0;        // 0 = tune default
    double seconds = 10.0;
    unsigned frequency = 48000;
    bool forceC64 = false;
    SidConfig::c64_model_t c64model = SidConfig::PAL;
    bool forceSid = false;
    SidConfig::sid_model_t sidmodel = SidConfig::MOS6581;
    std::string kernal, basic, chargen;
    // Fixed power-on delay for byte-exact determinism. libsidplayfp's default
    // (DEFAULT_POWER_ON_DELAY > MAX_POWER_ON_DELAY) draws a random delay from a
    // wall-clock-time-seeded RNG, which would make the cycle timeline differ
    // run to run. Pinning it (<= MAX_POWER_ON_DELAY) skips that path entirely.
    unsigned powerOnDelay = 0;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s [options] <file.sid>\n"
        "  --song N         subtune (1-based; 0=tune default)\n"
        "  --seconds S      emulated seconds to trace (default 10)\n"
        "  --out PREFIX     output prefix (writes PREFIX.bin, PREFIX.json)\n"
        "  --frequency HZ   SID sampling frequency (default 48000)\n"
        "  --model M        force c64 model: pal|ntsc|old-ntsc|drean|palm\n"
        "  --sid M          force sid model: 6581|8580\n"
        "  --power-on-delay N  fixed power-on delay cycles for determinism (default 0)\n"
        "  --kernal PATH    KERNAL ROM (8192 bytes; needed for most RSID)\n"
        "  --basic PATH     BASIC ROM (8192 bytes)\n"
        "  --chargen PATH   CHARGEN ROM (4096 bytes)\n",
        argv0);
}

bool readRom(const std::string &path, std::vector<uint8_t> &out, size_t expect) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open ROM %s\n", path.c_str()); return false; }
    out.resize(expect);
    size_t n = std::fread(out.data(), 1, expect, f);
    std::fclose(f);
    if (n != expect) {
        std::fprintf(stderr, "ROM %s: expected %zu bytes, got %zu\n", path.c_str(), expect, n);
        return false;
    }
    return true;
}

void jsonStr(std::string &o, const char *s) {
    o += '"';
    if (s) {
        for (const char *p = s; *p; ++p) {
            unsigned char c = static_cast<unsigned char>(*p);
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += static_cast<char>(c);
            }
        }
    }
    o += '"';
}

const char *clockName(SidTuneInfo::clock_t c) {
    switch (c) {
        case SidTuneInfo::CLOCK_PAL:  return "PAL";
        case SidTuneInfo::CLOCK_NTSC: return "NTSC";
        case SidTuneInfo::CLOCK_ANY:  return "ANY";
        default:                      return "UNKNOWN";
    }
}

// Nominal CPU clock (Hz) for the effective C64 model.
double cpuHz(SidConfig::c64_model_t m) {
    switch (m) {
        case SidConfig::NTSC:
        case SidConfig::OLD_NTSC: return 1022727.14;
        case SidConfig::DREAN:    return 1023440.0;
        case SidConfig::PAL_M:    return 1022727.14;
        case SidConfig::PAL:
        default:                  return 985248.444;
    }
}

} // namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs an argument\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--song") opt.song = std::strtoul(next("--song").c_str(), nullptr, 10);
        else if (a == "--seconds") opt.seconds = std::strtod(next("--seconds").c_str(), nullptr);
        else if (a == "--out") opt.prefix = next("--out");
        else if (a == "--frequency") opt.frequency = std::strtoul(next("--frequency").c_str(), nullptr, 10);
        else if (a == "--model") {
            std::string m = next("--model"); opt.forceC64 = true;
            if (m == "pal") opt.c64model = SidConfig::PAL;
            else if (m == "ntsc") opt.c64model = SidConfig::NTSC;
            else if (m == "old-ntsc") opt.c64model = SidConfig::OLD_NTSC;
            else if (m == "drean") opt.c64model = SidConfig::DREAN;
            else if (m == "palm") opt.c64model = SidConfig::PAL_M;
            else { std::fprintf(stderr, "unknown model %s\n", m.c_str()); return 2; }
        }
        else if (a == "--sid") {
            std::string m = next("--sid"); opt.forceSid = true;
            if (m == "6581") opt.sidmodel = SidConfig::MOS6581;
            else if (m == "8580") opt.sidmodel = SidConfig::MOS8580;
            else { std::fprintf(stderr, "unknown sid %s\n", m.c_str()); return 2; }
        }
        else if (a == "--power-on-delay") opt.powerOnDelay = std::strtoul(next("--power-on-delay").c_str(), nullptr, 10);
        else if (a == "--kernal") opt.kernal = next("--kernal");
        else if (a == "--basic") opt.basic = next("--basic");
        else if (a == "--chargen") opt.chargen = next("--chargen");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        else opt.input = a;
    }

    if (opt.input.empty()) { usage(argv[0]); return 2; }
    if (opt.prefix.empty()) {
        opt.prefix = opt.input;
        size_t dot = opt.prefix.rfind('.');
        if (dot != std::string::npos) opt.prefix.erase(dot);
        if (opt.song) opt.prefix += "_" + std::to_string(opt.song);
    }

    SidTune tune(opt.input.c_str());
    if (!tune.getStatus()) {
        std::fprintf(stderr, "failed to load %s: %s\n", opt.input.c_str(), tune.statusString());
        return 1;
    }
    tune.selectSong(opt.song);

    sidplayfp engine;

    std::vector<uint8_t> kbuf, bbuf, cbuf;
    const uint8_t *kp = nullptr, *bp = nullptr, *cp = nullptr;
    if (!opt.kernal.empty()) { if (!readRom(opt.kernal, kbuf, 8192)) return 1; kp = kbuf.data(); }
    if (!opt.basic.empty())  { if (!readRom(opt.basic,  bbuf, 8192)) return 1; bp = bbuf.data(); }
    if (!opt.chargen.empty()){ if (!readRom(opt.chargen, cbuf, 4096)) return 1; cp = cbuf.data(); }
    if (kp || bp || cp) engine.setRoms(kp, bp, cp);

    ReSIDfpBuilder builder("residfp");

    // Configure BEFORE load(). load() runs the power-on warm-up using the
    // engine's current config; if we configured after load(), that warm-up
    // would already have run with the default (random, time-seeded)
    // powerOnDelay, leaving a random absolute-cycle baseline. Setting the
    // fixed powerOnDelay first makes the single warm-up deterministic.
    SidConfig cfg = engine.config();
    cfg.frequency = opt.frequency;
    cfg.samplingMethod = SidConfig::INTERPOLATE;
    cfg.sidEmulation = &builder;
    cfg.forceC64Model = opt.forceC64;
    if (opt.forceC64) cfg.defaultC64Model = opt.c64model;
    cfg.forceSidModel = opt.forceSid;
    if (opt.forceSid) cfg.defaultSidModel = opt.sidmodel;
    // Force a deterministic power-on delay (see Options::powerOnDelay).
    cfg.powerOnDelay = static_cast<uint_least16_t>(opt.powerOnDelay & SidConfig::MAX_POWER_ON_DELAY);

    if (!engine.config(cfg)) {
        std::fprintf(stderr, "engine config failed: %s\n", engine.error());
        return 1;
    }

    if (!engine.load(&tune)) {
        std::fprintf(stderr, "engine load failed: %s\n", engine.error());
        return 1;
    }

    const SidTuneInfo *ti = tune.getInfo();
    const SidInfo &si = engine.info();

    // Effective C64 model (mirrors libsidplayfp model selection).
    SidConfig::c64_model_t eff = cfg.defaultC64Model;
    if (!opt.forceC64) {
        switch (ti->clockSpeed()) {
            case SidTuneInfo::CLOCK_NTSC: eff = SidConfig::NTSC; break;
            case SidTuneInfo::CLOCK_PAL:  eff = SidConfig::PAL;  break;
            default: break; // ANY/UNKNOWN keep default
        }
    }
    const double hz = cpuHz(eff);

    const std::string binPath = opt.prefix + ".bin";
    const std::string jsonPath = opt.prefix + ".json";

    FILE *bin = std::fopen(binPath.c_str(), "wb");
    if (!bin) { std::fprintf(stderr, "cannot open %s\n", binPath.c_str()); return 1; }

    TraceSink sink(bin);
    libsidplayfp::setInstrumentSink(&sink);

    const uint_least32_t targetMs = static_cast<uint_least32_t>(opt.seconds * 1000.0 + 0.5);
    const unsigned CHUNK = 10000;
    bool ok = true;
    while (engine.timeMs() < targetMs) {
        if (engine.play(CHUNK) < 0) {
            std::fprintf(stderr, "play error: %s\n", engine.error());
            ok = false;
            break;
        }
    }

    libsidplayfp::setInstrumentSink(nullptr);
    sink.flush();
    std::fclose(bin);

    // Metadata sidecar.
    auto info = [&](unsigned i) -> const char * {
        return (i < ti->numberOfInfoStrings()) ? ti->infoString(i) : "";
    };
    std::string j = "{\n";
    j += "  \"schema_version\": 1,\n";
    j += "  \"record_size\": 16,\n";
    j += "  \"num_records\": " + std::to_string(sink.count()) + ",\n";
    j += "  \"bin\": "; jsonStr(j, binPath.c_str()); j += ",\n";
    j += "  \"input\": "; jsonStr(j, opt.input.c_str()); j += ",\n";
    j += "  \"title\": "; jsonStr(j, info(0)); j += ",\n";
    j += "  \"author\": "; jsonStr(j, info(1)); j += ",\n";
    j += "  \"released\": "; jsonStr(j, info(2)); j += ",\n";
    j += "  \"format\": "; jsonStr(j, ti->formatString()); j += ",\n";
    j += "  \"speed_string\": "; jsonStr(j, si.speedString()); j += ",\n";
    j += "  \"songs\": " + std::to_string(ti->songs()) + ",\n";
    j += "  \"current_song\": " + std::to_string(ti->currentSong()) + ",\n";
    j += "  \"start_song\": " + std::to_string(ti->startSong()) + ",\n";
    j += "  \"load_addr\": " + std::to_string(ti->loadAddr()) + ",\n";
    j += "  \"init_addr\": " + std::to_string(ti->initAddr()) + ",\n";
    j += "  \"play_addr\": " + std::to_string(ti->playAddr()) + ",\n";
    j += "  \"clock_speed\": "; jsonStr(j, clockName(ti->clockSpeed())); j += ",\n";
    j += "  \"effective_model\": "; jsonStr(j, (eff==SidConfig::PAL?"PAL":eff==SidConfig::NTSC?"NTSC":eff==SidConfig::OLD_NTSC?"OLD_NTSC":eff==SidConfig::DREAN?"DREAN":"PAL_M")); j += ",\n";
    j += "  \"cpu_hz\": " + std::to_string(hz) + ",\n";
    j += "  \"num_sids\": " + std::to_string(si.numberOfSIDs()) + ",\n";
    j += "  \"sid_count\": " + std::to_string(ti->sidChips()) + ",\n";
    j += "  \"sid_base\": [";
    for (int i = 0; i < ti->sidChips(); ++i) {
        if (i) j += ", ";
        j += std::to_string(ti->sidChipBase(i));
    }
    j += "],\n";
    j += "  \"seconds\": " + std::to_string(opt.seconds) + ",\n";
    j += "  \"frequency\": " + std::to_string(opt.frequency) + ",\n";
    j += "  \"power_on_delay\": " + std::to_string(opt.powerOnDelay & SidConfig::MAX_POWER_ON_DELAY) + ",\n";
    j += "  \"deterministic\": true,\n";
    j += "  \"event_types\": {\"SID_WRITE\": 0, \"CIA_IRQ\": 1, \"VIC_IRQ\": 2, \"CPU_VECTOR\": 3}\n";
    j += "}\n";

    FILE *jf = std::fopen(jsonPath.c_str(), "wb");
    if (!jf) { std::fprintf(stderr, "cannot open %s\n", jsonPath.c_str()); return 1; }
    std::fwrite(j.data(), 1, j.size(), jf);
    std::fclose(jf);

    std::fprintf(stderr, "%s: %llu records -> %s (%s)\n", opt.input.c_str(),
                 static_cast<unsigned long long>(sink.count()), binPath.c_str(), jsonPath.c_str());
    return ok ? 0 : 1;
}
