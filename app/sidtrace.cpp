/*
 * sidtrace - byte-exact, cycle-stamped SID register + IRQ/NMI tracer with
 * program-state instrumentation for generator recovery.
 *
 * Part of preframr-playroutine. Built on libsidplayfp (GPL-2.0-or-later);
 * this program is therefore distributed under the GNU General Public License
 * version 2 or later.
 *
 * Plays a .sid exactly as sidplayfp would and emits (see docs/INSTRUMENTATION.md):
 *   <prefix>.bin       SID writes (now PC-tagged) + interrupts + CPU vectors
 *   <prefix>.ramwr.bin RAM write log (accumulators, table cursors, SMC)
 *   <prefix>.ramrd.bin RAM read log (only with --reads)
 *   <prefix>.cov.bin   executed-PC bitmap over play windows
 *   <prefix>.ram       64K RAM image (relocated player code + static tables)
 *   <prefix>.json      metadata sidecar
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

// Event record, 16 bytes, fields little-endian. Must match EVENT_DTYPE.
enum EventType : uint8_t {
    EV_SID_WRITE  = 0,  // chip=sid index, reg=0..0x1f, value, addr=full SID addr, aux=store-site PC
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
struct RamRec {
    uint64_t cycle;
    uint16_t pc;
    uint16_t addr;
    uint8_t  value;
    uint8_t  kind;
    uint16_t pad;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 16, "Record must be 16 bytes");
static_assert(sizeof(RamRec) == 16, "RamRec must be 16 bytes");

class TraceSink : public libsidplayfp::InstrumentSink {
public:
    TraceSink(FILE *ev, FILE *ramwr, FILE *ramrd, bool wantReads, uint8_t windowMask)
        : m_ev(ev), m_ramwr(ramwr), m_ramrd(ramrd),
          m_wantReads(wantReads), m_windowMask(windowMask) {
        m_evbuf.reserve(BUF);
        m_wrbuf.reserve(BUF);
        m_rdbuf.reserve(BUF);
        std::memset(m_cov, 0, sizeof(m_cov));
    }

    void sidWrite(int64_t cycle, int chip, uint16_t base,
                  uint8_t reg, uint8_t value, uint16_t pc) override {
        pushEv({static_cast<uint64_t>(cycle), EV_SID_WRITE,
                static_cast<uint8_t>(chip), reg, value,
                static_cast<uint16_t>(base + reg), pc});
    }

    void ciaIrq(int64_t cycle, int source, uint16_t taLatch, uint16_t tbLatch) override {
        pushEv({static_cast<uint64_t>(cycle), EV_CIA_IRQ,
                static_cast<uint8_t>(source), 0, 0, taLatch, tbLatch});
    }

    void vicIrq(int64_t cycle, unsigned rasterY, unsigned rasterCmp) override {
        pushEv({static_cast<uint64_t>(cycle), EV_VIC_IRQ, 3, 0, 0,
                static_cast<uint16_t>(rasterCmp), static_cast<uint16_t>(rasterY)});
    }

    void cpuVector(int64_t cycle, uint8_t kind, uint16_t pc) override {
        pushEv({static_cast<uint64_t>(cycle), EV_CPU_VECTOR, 0, 0, kind, pc, 0});
    }

    void ramWrite(int64_t cycle, uint16_t pc, uint16_t addr, uint8_t value, uint8_t kind) override {
        if (!windowWanted(kind) || m_ramwr == nullptr) return;
        m_wrbuf.push_back({static_cast<uint64_t>(cycle), pc, addr, value, kind, 0});
        ++m_nwr;
        if (m_wrbuf.size() >= BUF) flushBuf(m_wrbuf, m_ramwr);
    }

    void ramRead(int64_t cycle, uint16_t pc, uint16_t addr, uint8_t value, uint8_t kind) override {
        if (!windowWanted(kind) || m_ramrd == nullptr) return;
        m_rdbuf.push_back({static_cast<uint64_t>(cycle), pc, addr, value, kind, 0});
        ++m_nrd;
        if (m_rdbuf.size() >= BUF) flushBuf(m_rdbuf, m_ramrd);
    }

    void cpuExec(uint16_t pc, uint8_t kind) override {
        if (!windowWanted(kind)) return;
        m_cov[pc >> 3] |= static_cast<uint8_t>(1u << (pc & 7));
    }

    bool wantReads() const override { return m_wantReads; }

    void flush() {
        if (!m_evbuf.empty()) { std::fwrite(m_evbuf.data(), sizeof(Record), m_evbuf.size(), m_ev); m_evbuf.clear(); }
        flushBuf(m_wrbuf, m_ramwr);
        flushBuf(m_rdbuf, m_ramrd);
    }

    void writeCoverage(FILE *f) const { std::fwrite(m_cov, 1, sizeof(m_cov), f); }

    uint64_t evCount() const { return m_nev; }
    uint64_t wrCount() const { return m_nwr; }
    uint64_t rdCount() const { return m_nrd; }
    unsigned coverageCount() const {
        unsigned n = 0;
        for (unsigned char b : m_cov) n += __builtin_popcount(b);
        return n;
    }

private:
    static const size_t BUF = 65536;

    bool windowWanted(uint8_t kind) const { return (m_windowMask & (1u << kind)) != 0; }

    void pushEv(const Record &r) {
        m_evbuf.push_back(r);
        ++m_nev;
        if (m_evbuf.size() >= BUF) { std::fwrite(m_evbuf.data(), sizeof(Record), m_evbuf.size(), m_ev); m_evbuf.clear(); }
    }

    static void flushBuf(std::vector<RamRec> &buf, FILE *f) {
        if (f != nullptr && !buf.empty()) std::fwrite(buf.data(), sizeof(RamRec), buf.size(), f);
        buf.clear();
    }

    FILE *m_ev, *m_ramwr, *m_ramrd;
    bool m_wantReads;
    uint8_t m_windowMask;
    std::vector<Record> m_evbuf;
    std::vector<RamRec> m_wrbuf, m_rdbuf;
    uint8_t m_cov[8192];
    uint64_t m_nev = 0, m_nwr = 0, m_nrd = 0;
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
    unsigned powerOnDelay = 0;   // fixed for byte-exact determinism (see README)
    bool reads = false;
    bool ramwrites = true;
    bool coverage = true;
    bool ramimage = true;
    uint8_t windowMask = 0x3;    // bit0=IRQ, bit1=NMI; default both
    double ramDumpSeconds = 0.0; // dump RAM image at first play window after init
};

void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s [options] <file.sid>\n"
        "  --song N            subtune (1-based; 0=tune default)\n"
        "  --seconds S         emulated seconds to trace (default 10)\n"
        "  --out PREFIX        output prefix\n"
        "  --frequency HZ      SID sampling frequency (default 48000)\n"
        "  --model M           force c64 model: pal|ntsc|old-ntsc|drean|palm\n"
        "  --sid M             force sid model: 6581|8580\n"
        "  --power-on-delay N  fixed power-on delay cycles for determinism (default 0)\n"
        "  --window W          play-window source: irq|nmi|both (default both)\n"
        "  --reads             also emit the RAM read log (large)\n"
        "  --no-ramwrites      do not emit the RAM write log\n"
        "  --no-coverage       do not emit the PC coverage bitmap\n"
        "  --no-ram            do not dump the RAM image\n"
        "  --ram-dump-seconds S  emulated time to capture the RAM image (default 0)\n"
        "  --kernal/--basic/--chargen PATH   ROM images (RSID needs KERNAL)\n",
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
                    // Escape control and non-ASCII bytes as \u00XX so the JSON
                    // is valid UTF-8 (SID metadata strings are Windows-1252).
                    if (c < 0x20 || c >= 0x7f) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
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
        else if (a == "--window") {
            std::string w = next("--window");
            if (w == "irq") opt.windowMask = 0x1;
            else if (w == "nmi") opt.windowMask = 0x2;
            else if (w == "both") opt.windowMask = 0x3;
            else { std::fprintf(stderr, "unknown window %s\n", w.c_str()); return 2; }
        }
        else if (a == "--reads") opt.reads = true;
        else if (a == "--no-ramwrites") opt.ramwrites = false;
        else if (a == "--no-coverage") opt.coverage = false;
        else if (a == "--no-ram") opt.ramimage = false;
        else if (a == "--ram-dump-seconds") opt.ramDumpSeconds = std::strtod(next("--ram-dump-seconds").c_str(), nullptr);
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

    // Configure BEFORE load() so the single power-on warm-up uses the fixed
    // (deterministic) powerOnDelay rather than the default random one.
    SidConfig cfg = engine.config();
    cfg.frequency = opt.frequency;
    cfg.samplingMethod = SidConfig::INTERPOLATE;
    cfg.sidEmulation = &builder;
    cfg.forceC64Model = opt.forceC64;
    if (opt.forceC64) cfg.defaultC64Model = opt.c64model;
    cfg.forceSidModel = opt.forceSid;
    if (opt.forceSid) cfg.defaultSidModel = opt.sidmodel;
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

    SidConfig::c64_model_t eff = cfg.defaultC64Model;
    if (!opt.forceC64) {
        switch (ti->clockSpeed()) {
            case SidTuneInfo::CLOCK_NTSC: eff = SidConfig::NTSC; break;
            case SidTuneInfo::CLOCK_PAL:  eff = SidConfig::PAL;  break;
            default: break;
        }
    }
    const double hz = cpuHz(eff);

    const std::string binPath = opt.prefix + ".bin";
    const std::string ramwrPath = opt.prefix + ".ramwr.bin";
    const std::string ramrdPath = opt.prefix + ".ramrd.bin";
    const std::string covPath = opt.prefix + ".cov.bin";
    const std::string ramPath = opt.prefix + ".ram";
    const std::string jsonPath = opt.prefix + ".json";

    FILE *bin = std::fopen(binPath.c_str(), "wb");
    if (!bin) { std::fprintf(stderr, "cannot open %s\n", binPath.c_str()); return 1; }
    FILE *ramwr = opt.ramwrites ? std::fopen(ramwrPath.c_str(), "wb") : nullptr;
    FILE *ramrd = opt.reads ? std::fopen(ramrdPath.c_str(), "wb") : nullptr;

    TraceSink sink(bin, ramwr, ramrd, opt.reads, opt.windowMask);
    libsidplayfp::setInstrumentSink(&sink);

    const uint_least32_t targetMs = static_cast<uint_least32_t>(opt.seconds * 1000.0 + 0.5);
    const uint_least32_t dumpMs = static_cast<uint_least32_t>(opt.ramDumpSeconds * 1000.0 + 0.5);
    const unsigned CHUNK = 10000;
    bool ok = true;
    bool ramDumped = false;
    std::vector<uint8_t> ramImg(0x10000);
    uint64_t ramDumpCycle = 0;

    while (engine.timeMs() < targetMs) {
        if (engine.play(CHUNK) < 0) {
            std::fprintf(stderr, "play error: %s\n", engine.error());
            ok = false;
            break;
        }
        if (!ramDumped && engine.timeMs() >= dumpMs) {
            engine.getRam(ramImg.data());
            ramDumped = true;
            ramDumpCycle = static_cast<uint64_t>(engine.timeMs());
        }
    }
    if (!ramDumped) { engine.getRam(ramImg.data()); ramDumpCycle = static_cast<uint64_t>(engine.timeMs()); }

    libsidplayfp::setInstrumentSink(nullptr);
    sink.flush();
    std::fclose(bin);
    if (ramwr) std::fclose(ramwr);
    if (ramrd) std::fclose(ramrd);

    if (opt.coverage) {
        FILE *cov = std::fopen(covPath.c_str(), "wb");
        if (cov) { sink.writeCoverage(cov); std::fclose(cov); }
    }
    if (opt.ramimage) {
        FILE *rf = std::fopen(ramPath.c_str(), "wb");
        if (rf) { std::fwrite(ramImg.data(), 1, ramImg.size(), rf); std::fclose(rf); }
    }

    auto info = [&](unsigned i) -> const char * {
        return (i < ti->numberOfInfoStrings()) ? ti->infoString(i) : "";
    };
    const char *windowStr = opt.windowMask == 0x1 ? "irq" : opt.windowMask == 0x2 ? "nmi" : "both";

    std::string j = "{\n";
    j += "  \"schema_version\": 2,\n";
    j += "  \"record_size\": 16,\n";
    j += "  \"num_records\": " + std::to_string(sink.evCount()) + ",\n";
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
    for (int i = 0; i < ti->sidChips(); ++i) { if (i) j += ", "; j += std::to_string(ti->sidChipBase(i)); }
    j += "],\n";
    j += "  \"seconds\": " + std::to_string(opt.seconds) + ",\n";
    j += "  \"frequency\": " + std::to_string(opt.frequency) + ",\n";
    j += "  \"power_on_delay\": " + std::to_string(opt.powerOnDelay & SidConfig::MAX_POWER_ON_DELAY) + ",\n";
    j += "  \"deterministic\": true,\n";
    j += "  \"window\": "; jsonStr(j, windowStr); j += ",\n";
    j += "  \"reads_enabled\": "; j += (opt.reads ? "true" : "false"); j += ",\n";
    j += "  \"ram_dump_cycle\": " + std::to_string(ramDumpCycle) + ",\n";
    j += "  \"num_ram_writes\": " + std::to_string(sink.wrCount()) + ",\n";
    j += "  \"num_ram_reads\": " + std::to_string(sink.rdCount()) + ",\n";
    j += "  \"coverage_count\": " + std::to_string(sink.coverageCount()) + ",\n";
    j += "  \"artifacts\": {";
    j += "\"sidwr\": "; jsonStr(j, binPath.c_str());
    if (opt.ramwrites) { j += ", \"ramwr\": "; jsonStr(j, ramwrPath.c_str()); }
    if (opt.reads)     { j += ", \"ramrd\": "; jsonStr(j, ramrdPath.c_str()); }
    if (opt.coverage)  { j += ", \"cov\": "; jsonStr(j, covPath.c_str()); }
    if (opt.ramimage)  { j += ", \"ram\": "; jsonStr(j, ramPath.c_str()); }
    j += "},\n";
    j += "  \"ramacc_fields\": [\"cycle\", \"pc\", \"addr\", \"value\", \"kind\", \"pad\"],\n";
    j += "  \"event_types\": {\"SID_WRITE\": 0, \"CIA_IRQ\": 1, \"VIC_IRQ\": 2, \"CPU_VECTOR\": 3},\n";
    j += "  \"window_kinds\": {\"IRQ\": 0, \"NMI\": 1}\n";
    j += "}\n";

    FILE *jf = std::fopen(jsonPath.c_str(), "wb");
    if (!jf) { std::fprintf(stderr, "cannot open %s\n", jsonPath.c_str()); return 1; }
    std::fwrite(j.data(), 1, j.size(), jf);
    std::fclose(jf);

    std::fprintf(stderr, "%s: %llu events, %llu ramwr, cov=%u -> %s\n", opt.input.c_str(),
                 static_cast<unsigned long long>(sink.evCount()),
                 static_cast<unsigned long long>(sink.wrCount()),
                 sink.coverageCount(), binPath.c_str());
    return ok ? 0 : 1;
}
