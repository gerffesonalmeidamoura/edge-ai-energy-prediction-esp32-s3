#include <SPI.h>
#include <SD.h>
#include <ADE7880.h>
#include <math.h>

// ===================== Configuráveis =====================
#define START_DAY    6     // <- defina a data inicial da 1ª coleta
#define START_MONTH   11
#define START_YEAR  2025

// Hora/minuto iniciais da 1ª coleta do dia (definidos pelo usuário)
#define START_HOUR    21   // ex.: 00:16 -> START_HOUR 0, START_MINUTE 16
#define START_MINUTE  49

// LEDs e tempo
#define LED_PIN        2
#define BLINK_ON_MS   80
#define BLINK_OFF_MS  80

// ===================== Pinos / SPI SD =====================
#define CS_PIN    5    // CS do SD
#define PIN_SCK  18
#define PIN_MISO 19
#define PIN_MOSI 23

// ===================== ADE7880 / Constantes =====================
#define HREADY_BIT 19
const float VLSB  = 0.00006716f;
const float ILSB  = 0.0000055f;
const float PLSB  = 0.01f;
const float WLSB  = 0.00000002307f;
const float SaLSB = 0.01f;
const float CLSB  = 0.0000305f;

ADE7880_I2C eic;
bool dsp_started = false;

// ===================== Variáveis de medição =====================
float urms_l1=0, Irms_l1=0, freq_l1=0, Pwatt_l1=0, Qvar_l1=0, Sva_l1=0, cosfL1=0;
double energia_acumulada = 0;
float v_fundamental_l1=0, i_fundamental_l1=0, thd_v_l1=0, thd_i_l1=0;
float Vharmonics[51], Iharmonics[51];

// ===================== Controle por dia =====================
uint16_t contador_medidoes_dia = 0;
bool primeira_amostra_ok = false;   // agora indica que a 1ª passagem já ocorreu (mesmo sem gravar)

// Data corrente (persistida)
uint16_t cur_day   = START_DAY;
uint16_t cur_month = START_MONTH;
uint16_t cur_year  = START_YEAR;

// Nomes de arquivos
char caminho_temp[32];
char caminho_final[48];

// ===================== Helpers LED =====================
void blinkOK(uint8_t times = 3) {
  for (uint8_t i=0;i<times;i++){
    digitalWrite(LED_PIN, HIGH); delay(BLINK_ON_MS);
    digitalWrite(LED_PIN, LOW);  delay(BLINK_OFF_MS);
  }
}
void blinkERROR(uint8_t times = 2) {
  for (uint8_t i=0;i<times;i++){
    digitalWrite(LED_PIN, HIGH); delay(220);
    digitalWrite(LED_PIN, LOW);  delay(220);
  }
}

// ===================== Data / Calendário =====================
bool isLeap(uint16_t y){ return ((y%4==0 && y%100!=0) || (y%400==0)); }
uint8_t daysInMonth(uint16_t m, uint16_t y){
  switch(m){
    case 1: case 3: case 5: case 7: case 8: case 10: case 12: return 31;
    case 4: case 6: case 9: case 11: return 30;
    case 2: return isLeap(y)?29:28;
    default: return 30;
  }
}
void incDateOneDay(uint16_t &d, uint16_t &m, uint16_t &y){
  uint8_t dim = daysInMonth(m,y);
  if (++d > dim){ d = 1; if(++m>12){ m=1; y++; } }
}
void formatDateCompact(char out[7], uint16_t d, uint16_t m, uint16_t y){
  // ddmmyy
  uint8_t yy = (uint8_t)(y % 100);
  snprintf(out, 7, "%02u%02u%02u", (unsigned)d, (unsigned)m, (unsigned)yy);
}
void makeTempPath(char out[32], uint16_t d, uint16_t m, uint16_t y){
  char dc[7]; formatDateCompact(dc,d,m,y);
  snprintf(out, 32, "/dia%s_temp.csv", dc);
}
void makeFinalPath(char out[48], uint16_t d, uint16_t m, uint16_t y, float energia_final){
  char dc[7]; formatDateCompact(dc,d,m,y);
  // Ex.: /dia130925_10.4691.csv
  snprintf(out, 48, "/dia%s_%.4f.csv", dc, energia_final);
}

// ===================== Helpers HH:MM (1ª coluna) =====================
// Converte índice de medição (0..1439, ... 1440 == volta) para HH:MM a partir de START_HOUR/MINUTE
void indexToTimeHM(uint16_t idx, char out[6]){
  const uint32_t base = (uint32_t)START_HOUR * 60u + (uint32_t)START_MINUTE;
  const uint32_t total = (base + (uint32_t)idx) % 1440u;
  const uint8_t  hh = total / 60u;
  const uint8_t  mm = total % 60u;
  snprintf(out, 6, "%02u:%02u", (unsigned)hh, (unsigned)mm);
}

// Faz o inverso: dada "HH:MM", obtém o índice relativo ao START_HOUR/MINUTE.
int16_t timeHMToIndex(uint8_t hh, uint8_t mm){
  const int16_t base = (int16_t)START_HOUR * 60 + (int16_t)START_MINUTE;
  const int16_t now  = (int16_t)hh * 60 + (int16_t)mm;
  // diferença modular em [0, 1439]
  int16_t diff = now - base;
  while (diff < 0)     diff += 1440;
  while (diff >= 1440) diff -= 1440;
  return diff;
}

// ===================== SD helpers =====================
bool initSDWithRetry(uint8_t attempts = 4) {
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, CS_PIN);

  const uint32_t freqs[] = {400000UL, 1000000UL, 4000000UL, 8000000UL};
  for (uint8_t a=0; a<attempts; ++a){
    for (uint8_t i=0;i<sizeof(freqs)/sizeof(freqs[0]);++i){
      SD.end(); delay(10);
      if (SD.begin(CS_PIN, SPI, freqs[i])) {
        // Prova de R/W
        File probe = SD.open("/.__probe__.txt", FILE_WRITE);
        if (probe){
          probe.println("ok");
          probe.flush();
          probe.close();
          SD.remove("/.__probe__.txt");
          return true;
        }
      }
    }
    delay(60 + 40*a);
  }
  return false;
}

bool arquivoExiste(const char* path) {
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  f.close();
  return true;
}

bool lerUltimaLinha(const char* path, String &linha) {
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  int32_t pos = f.size() - 1;
  if (pos < 0) { f.close(); return false; }

  f.seek(pos);
  if (f.read()=='\n') pos--;
  while (pos>=0){
    f.seek(pos);
    if (f.read()=='\n'){ pos++; break; }
    pos--;
  }
  if (pos<0) pos = 0;

  f.seek(pos);
  linha = "";
  while (f.available()){
    char c = f.read();
    if (c=='\n' || c=='\r') break;
    linha += c;
  }
  f.close();
  return linha.length()>0;
}

// ========== Parser compatível: HH:MM (novo) ou milissegundos (legado) ==========
bool parseTimestampEnergia(const String &linha, uint32_t &ts_idx_min, double &energia_kwh) {
  // Primeiro campo até a primeira vírgula
  int firstComma = linha.indexOf(',');
  if (firstComma < 0) return false;
  String firstField = linha.substring(0, firstComma);
  firstField.trim();

  // Último campo (Energia)
  int lastComma = linha.lastIndexOf(',');
  if (lastComma < 0) return false;
  energia_kwh = strtod(linha.substring(lastComma+1).c_str(), nullptr);

  // 1) Formato novo: "HH:MM"
  int colon = firstField.indexOf(':');
  if (colon >= 0) {
    int hh = firstField.substring(0, colon).toInt();
    int mm = firstField.substring(colon+1).toInt();
    if (hh < 0 || hh > 23 || mm < 0 || mm > 59) return false;
    int16_t idx = timeHMToIndex((uint8_t)hh, (uint8_t)mm);
    if (idx < 0 || idx > 1439) return false;
    ts_idx_min = (uint32_t)idx;
    return true;
  }

  // 2) Formato legado: milissegundos (0, 60000, 120000, ...)
  char *endp = nullptr;
  unsigned long ts_ms = strtoul(firstField.c_str(), &endp, 10);
  if (endp == firstField.c_str()) return false;
  ts_idx_min = (uint32_t)(ts_ms / 60000UL);
  return true;
}

// ===================== Cabeçalho CSV (NOVO helper) =====================
static inline void writeCSVHeader(File &f){
  f.print("timestamp,Urms_L1,Irms_L1,Frequencia_L1,Pwatt_L1,Qvar_L1,Sva_L1,CosF_L1,");
  f.print("TensaoFundamental_L1,CorrenteFundamental_L1,THD_V_L1,THD_I_L1");
  for (int i=0;i<=50;i++){ f.print(",UH"); f.print(i); }
  for (int i=0;i<=50;i++){ f.print(",IH"); f.print(i); }
  f.println(",EnergiaAcumulada_KWh");
  f.flush();
}

// ===================== Garantia de cabeçalho (NOVO) =====================
bool ensureHeaderPresent(const char* path){
  if (!arquivoExiste(path)) return false;

  File fin = SD.open(path, FILE_READ);
  if (!fin) return false;

  size_t fsize = fin.size();
  String firstLine = (fsize > 0) ? fin.readStringUntil('\n') : String("");
  fin.close();
  firstLine.trim();

  bool hasHeader = (firstLine.indexOf("Urms_L1") >= 0);
  if (hasHeader) return false; // já tem cabeçalho

  // Caso 1: arquivo existe mas está vazio -> escreve cabeçalho direto
  if (fsize == 0){
    File f = SD.open(path, FILE_WRITE);
    if (!f) return false;
    writeCSVHeader(f);
    f.close();
    Serial.print("Header inserido (arquivo vazio): "); Serial.println(path);
    return true;
  }

  // Caso 2: arquivo tem dados sem cabeçalho -> cria novo, injeta cabeçalho e copia dados
  const char* TMP = "/.__hdr_fix__.csv";
  SD.remove(TMP);

  File fout = SD.open(TMP, FILE_WRITE);
  if (!fout) return false;
  writeCSVHeader(fout);

  fin = SD.open(path, FILE_READ);
  if (!fin){ fout.close(); SD.remove(TMP); return false; }

  while (fin.available()){
    int c = fin.read();
    if (c < 0) break;
    fout.write((uint8_t)c);
  }
  fin.close();
  fout.flush();
  fout.close();

  SD.remove(path);
  SD.rename(TMP, path);

  Serial.print("Header inserido (arquivo migrado): "); Serial.println(path);
  return true;
}

// Abre/cria e appenda uma linha (com cabeçalho, se novo). Retenta remonte pontual.
bool appendCSVLine(const char* path, const String& linha) {
  // >>> NOVO: garante que haverá cabeçalho no arquivo (cobre casos antigos e novos)
  if (SD.cardType()){
    ensureHeaderPresent(path);
  }

  for (int tent=0; tent<3; ++tent){
    File f = SD.open(path, FILE_APPEND);
    if (!f) f = SD.open(path, FILE_WRITE);
    if (!f){
      // remonta rápido e tenta de novo
      SD.end(); delay(20);
      initSDWithRetry(1);
      delay(20);
      continue;
    }

    if (f.size()==0){
      // arquivo recém-criado -> escreve cabeçalho
      writeCSVHeader(f);
    }

    size_t n = f.print(linha);
    f.flush();
    f.close();
    return (n == linha.length());
  }
  return false;
}

// ===================== Estado persistente =====================
// /state.txt: d,m,y,contador,energia
void saveState(uint16_t d, uint16_t m, uint16_t y, uint16_t contador, double energia){
  File f = SD.open("/state.txt", FILE_WRITE);
  if (!f) return;
  f.seek(0);
  f.print("dia=");      f.println(d);
  f.print("mes=");      f.println(m);
  f.print("ano=");      f.println(y);
  f.print("contador="); f.println(contador);
  f.print("energia=");  f.println(energia, 8);
  f.flush();
  f.close();
}

bool loadState(uint16_t &d, uint16_t &m, uint16_t &y, uint16_t &contador, double &energia){
  File f = SD.open("/state.txt", FILE_READ);
  if (!f) return false;
  String sD,sM,sY,sC,sE;
  while (f.available()){
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.startsWith("dia="))      sD = line.substring(4);
    else if (line.startsWith("mes=")) sM = line.substring(4);
    else if (line.startsWith("ano=")) sY = line.substring(4);
    else if (line.startsWith("contador=")) sC = line.substring(9);
    else if (line.startsWith("energia="))  sE = line.substring(8);
  }
  f.close();
  if (!sD.length() || !sM.length() || !sY.length()) return false;
  d = (uint16_t) strtoul(sD.c_str(), nullptr, 10);
  m = (uint16_t) strtoul(sM.c_str(), nullptr, 10);
  y = (uint16_t) strtoul(sY.c_str(), nullptr, 10);
  contador = (uint16_t) strtoul(sC.c_str(), nullptr, 10);
  energia  = strtod(sE.c_str(), nullptr);
  return true;
}

// ===================== Harmônicas com timeout =====================
void lerHarmonicasComTimeout() {
  for (int k=1;k<=51;++k){
    int idx = k-1;
    Vharmonics[idx] = NAN;
    Iharmonics[idx] = NAN;

    eic.write8Register(HX_reg, k);
    delay(4);

    const uint16_t tentativas = 5; // máx ~250 ms por banda
    bool ok=false;
    for (uint16_t t=0;t<tentativas;++t){
      uint32_t status0 = eic.read32Register(STATUS0);
      if (status0 & (1UL << HREADY_BIT)){
        float vh = eic.read32Register(HXVRMS) * VLSB;
        float ih = eic.read32Register(HXIRMS) * ILSB;
        Vharmonics[idx] = (k==2) ? vh : (vh*0.01f);
        Iharmonics[idx] = ih;
        ok=true; break;
      }
      delay(50);
    }
    // se !ok, mantém NAN e segue (não bloqueia)
  }
}

// ===================== Validação de leitura =====================
static inline bool valorValido(float x){ return isfinite(x) && (x>-1e30f) && (x<1e30f); }
bool leituraBaseValida(){
  // parâmetros mínimos para considerar “ok”
  if (!valorValido(urms_l1) || urms_l1 < 10.0f) return false;  // tensão precisa “existir”
  if (!valorValido(Irms_l1)) return false;
  if (!valorValido(freq_l1) || freq_l1 < 40.0f || freq_l1 > 70.0f) return false;
  if (!valorValido(Pwatt_l1) || !valorValido(Sva_l1) || !valorValido(cosfL1)) return false;
  return true;
}

// Tenta obter uma leitura base confiável dentro de um tempo limite.
// Retorna true se conseguiu; caso contrário, false (o loop não trava).
bool tentarLerBaseComRetentativas(uint32_t timeout_ms = 2500){
  uint32_t tStart = millis();
  uint8_t  miniResets = 0;

  while ((millis()-tStart) < timeout_ms){
    // leituras base
    urms_l1 = eic.read32Register(AVRMS) * VLSB;
    Irms_l1 = eic.read32Register(AIRMS) * ILSB;
    freq_l1 = eic.read32Register(APERIOD) * 0.00000021448f;

    double energia_atual = eic.read32Register(AWATTHR) * WLSB;
    if (isfinite(energia_atual) && energia_atual >= 0.0) {
      energia_acumulada += energia_atual;
    }

    Pwatt_l1 = eic.read32Register(AWATT) * PLSB;
    Sva_l1   = eic.read32Register(AVA)   * SaLSB;
    cosfL1   = eic.read16Register(APF)   * CLSB;
    Qvar_l1  = Sva_l1 * sqrtf(max(0.0f, 1.0f - cosfL1 * cosfL1));
    v_fundamental_l1 = eic.read32Register(FVRMS) * VLSB;
    i_fundamental_l1 = eic.read32Register(FIRMS) * ILSB;
    thd_v_l1 = eic.read32Register(VTHDN) * VLSB * 0.094f;
    thd_i_l1 = eic.read32Register(ITHDN) * ILSB * 9.665f;

    if (leituraBaseValida()) return true;

    // Se ainda não validou, dá um "kick" no DSP e tenta de novo (poucas vezes)
    if (miniResets < 2){
      eic.runDsp();
      miniResets++;
    }
    delay(60);
  }
  return false;
}

// ===================== Recuperação de progresso =====================
void recuperarProgresso(){
  // tenta SD (não bloquear aqui)
  initSDWithRetry(4);

  // tenta carregar estado
  uint16_t d,m,y,cont;
  double   en;
  if (loadState(d,m,y,cont,en)){
    cur_day = d; cur_month = m; cur_year = y;
    contador_medidoes_dia = cont;
    energia_acumulada     = en;
    makeTempPath(caminho_temp, cur_day, cur_month, cur_year);

    // se existir temp, tenta recuperar a partir da última linha
    if (arquivoExiste(caminho_temp)){
      String last;
      if (lerUltimaLinha(caminho_temp, last)){
        uint32_t ts; double e_kwh;
        if (parseTimestampEnergia(last, ts, e_kwh)){
          // ts agora é o índice de minuto (0..1439) para novo formato,
          // ou convertido a partir de ms (legado)
          contador_medidoes_dia = (uint16_t)ts + 1;
          energia_acumulada = e_kwh;
        }
      }
    } else {
      // não tinha temp do dia corrente → começará um novo arquivo do dia corrente
    }
  } else {
    // sem state → usa data inicial de projeto e zera progresso
    cur_day = START_DAY; cur_month = START_MONTH; cur_year = START_YEAR;
    contador_medidoes_dia = 0;
    energia_acumulada = 0;
  }

  // atualiza caminho temp
  makeTempPath(caminho_temp, cur_day, cur_month, cur_year);

  // (opcional) poderíamos chamar ensureHeaderPresent aqui, mas a chamada em append já cobre tudo.
}

// ===================== Setup =====================
void setup(){
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(150);

  // Não bloquear: tenta SD, mas segue mesmo se falhar
  initSDWithRetry(6);

  recuperarProgresso();

  // ADE7880 init
  eic.begin();
  delay(100);
  eic.write16Register(HCONFIG, 0x8001);   delay(200);
  eic.write8Register(LCYCMODE, 0x40);    delay(250);
  eic.write8Register(WTHR, 3);
  eic.write8Register(VARTHR, 3);
  eic.write8Register(VATHR, 3);
  eic.write16Register(COMPMODE, 0x0008);
  eic.write32Register(VLEVEL, 0x38000);
  eic.write32Register(AIGAIN, 0xFC735F);
  eic.write32Register(BIGAIN, 0x000000);
  eic.write32Register(CIGAIN, 0x000000);
  eic.write32Register(AVGAIN, 0x000000);
  eic.write32Register(BVGAIN, 0x000000);
  eic.write32Register(CVGAIN, 0x000000);
  eic.write32Register(APGAIN, 0x000000);
  eic.write32Register(BPGAIN, 0x000000);
  eic.write32Register(CPGAIN, 0x000000);
  eic.write32Register(AWATTOS, 0x000000);
  eic.write32Register(BWATTOS, 0x000000);
  eic.write32Register(CWATTOS, 0x000000);
  eic.write32Register(AIRMSOS, 0x0001CE8);
  eic.write16Register(VTHDN, 0x0001);
  eic.write16Register(ITHDN, 0x0001);

  uint16_t compmode = eic.read16Register(COMPMODE);
  compmode |= (1 << 14);
  eic.write16Register(COMPMODE, compmode);
  delay(80);

  Serial.println("Setup concluído (sem bloqueios).");
}

// ===================== Loop =====================
void loop(){
  const unsigned long t0 = millis();

  if (!dsp_started){
    eic.runDsp();
    dsp_started = true;
    Serial.println("DSP iniciado!");
  }

  // 1) Garante leitura base com retentativas/timeout (não trava)
  bool gotBase = tentarLerBaseComRetentativas(3000);
  if (!gotBase){
    Serial.println("Aviso: leitura base não validou no tempo. Continuando para não travar.");
  }

  // 2) Harmônicas
  if (primeira_amostra_ok){
    lerHarmonicasComTimeout();
  } else {
    // Preenche com NANs na primeira passagem para manter formato (mesmo descartando)
    for (int i=0;i<=50;i++){ Vharmonics[i]=NAN; Iharmonics[i]=NAN; }
  }

  // 3) Monta linha (1ª coluna em HH:MM)
  char hhmm[6];
  indexToTimeHM(contador_medidoes_dia, hhmm);

  String linha; linha.reserve(1150);
  linha += hhmm; linha += ",";
  linha += String(urms_l1,4) + "," + String(Irms_l1,4) + "," + String(freq_l1,4) + ",";
  linha += String(Pwatt_l1,4) + "," + String(Qvar_l1,4) + "," + String(Sva_l1,4) + ",";
  linha += String(cosfL1,4) + "," + String(v_fundamental_l1,4) + "," + String(i_fundamental_l1,4) + ",";
  linha += String(thd_v_l1,4) + "," + String(thd_i_l1,4);

  for (int i=0;i<=50;i++){ linha += ","; linha += String(Vharmonics[i],5); }
  for (int i=0;i<=50;i++){ linha += ","; linha += String(Iharmonics[i],5); }

  linha += ","; linha += String(energia_acumulada,4); linha += "\n";

  // 4) Garante SD montado (sem travar): se falhou antes, tenta de novo agora
  if (!SD.cardType()){
    initSDWithRetry(3);
  }

  // ===== NOVO: descartar a 1ª amostra (não gravar) =====
  bool should_write = primeira_amostra_ok; // false na 1ª passagem, true depois

  // 5) Gravação robusta (apenas se não for a 1ª amostra)
  if (should_write){
    bool gravado = false;
    if (SD.cardType()){
      gravado = appendCSVLine(caminho_temp, linha);
    }
    if (!gravado){
      Serial.println("ERRO: não foi possível gravar a linha (SD indisponível ou falha de I/O).");
      blinkERROR(); // 2 piscadas longas = falha
    } else {
      blinkOK(3);   // 3 piscadas curtas = sucesso

      // Atualiza estado somente após gravação OK
      saveState(cur_day, cur_month, cur_year, contador_medidoes_dia, energia_acumulada);

      Serial.print("Gravado: medição ");
      Serial.print(contador_medidoes_dia);
      Serial.print(" ("); Serial.print(hhmm); Serial.print(") do dia ");
      Serial.print(cur_day); Serial.print("/");
      Serial.print(cur_month); Serial.print("/");
      Serial.println(cur_year);
    }
  } else {
    // Primeira passagem: não grava, só marca estabilização e informa
    Serial.print("Descartando 1ª amostra para estabilização em ");
    Serial.println(hhmm);
    primeira_amostra_ok = true;

    // Opcional: salvar estado indicando que já passamos pela 1ª amostra
    saveState(cur_day, cur_month, cur_year, contador_medidoes_dia, energia_acumulada);
  }

  // Avança o índice de minuto SEMPRE (gravando ou não)
  contador_medidoes_dia++;

  // 6) Final de dia
  // Observação: como descartamos o índice 0, ainda assim teremos 1440 gravações
  // porque gravamos de 1..1440 (o índice 1440 rotaciona para o horário inicial).
  if (contador_medidoes_dia >= 1440){
    float energia_final = (float)energia_acumulada;

    // Gera caminho final com data
    makeFinalPath(caminho_final, cur_day, cur_month, cur_year, energia_final);

    // Tenta renomear temp->final (se SD ok)
    if (SD.cardType()){
      if (SD.rename(caminho_temp, caminho_final)){
        Serial.print("Dia concluído -> "); Serial.println(caminho_final);
      } else {
        Serial.println("Falha ao renomear. Mantendo arquivo _temp.csv.");
      }
    } else {
      Serial.println("SD indisponível ao fim do dia. Arquivo _temp permanece.");
    }

    // Prepara próximo dia
    incDateOneDay(cur_day, cur_month, cur_year);
    contador_medidoes_dia = 0;
    energia_acumulada = 0.0;
    primeira_amostra_ok = false;

    // Atualiza state e novo temp
    saveState(cur_day, cur_month, cur_year, contador_medidoes_dia, energia_acumulada);
    makeTempPath(caminho_temp, cur_day, cur_month, cur_year);
    Serial.println("----------- NOVO DIA INICIADO -----------");
  }

  // 7) Heartbeat simples (toggle)
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));

  // 8) Controle de período ~60s (respeitando o tempo gasto)
  const unsigned long periodo = 60000UL;
  unsigned long elapsed = millis() - t0;
  if (elapsed < periodo) delay(periodo - elapsed);

  // Log
  Serial.print("Tempo decorrido (ms): ");
  Serial.println((unsigned long)(millis() - t0));
}
