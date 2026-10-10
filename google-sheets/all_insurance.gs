// ==========================================================================
// ALL INSURANCE - PROPENSÃO DE COMPRA DO SEGURO AUTOMOTIVO
//
// Envia os novos clientes da planilha para a API do modelo, escreve o score
// de propensão de compra de cada cliente e ordena a lista do maior para o
// menor score, para que o time de vendas ligue primeiro para os clientes
// com mais chance de contratar o seguro automotivo.
// ==========================================================================

// URL da API publicada no Heroku (substitua pela URL do seu app)
const API_URL = 'https://all-insurance-api-163fa9a62447.herokuapp.com';

// quantidade de clientes enviados em cada requisição
const TAMANHO_LOTE = 1000;

// colunas que a API precisa receber: apenas elas são enviadas
const COLUNAS_OBRIGATORIAS = ['gender', 'age', 'region_code', 'driving_license', 'previously_insured',
                              'annual_premium', 'policy_sales_channel', 'vehicle_age', 'vehicle_damage'];

// colunas criadas pelo script
const COLUNA_SCORE = 'score';
const COLUNA_RANKING = 'ranking';

// cor de destaque dos clientes que devem receber a ligação
const COR_DESTAQUE = '#d6e2f0';


// cria o menu da All Insurance quando a planilha é aberta
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('All Insurance')
    .addItem('Calcular propensão de compra', 'calcularPropensao')
    .addItem('Destacar clientes para ligação', 'destacarClientes')
    .addToUi();
}


function calcularPropensao() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const dados = sheet.getDataRange().getValues();
  const cabecalho = dados[0].map(String);

  // verifica se a planilha tem todas as colunas que a API precisa
  const colunasAusentes = COLUNAS_OBRIGATORIAS.filter(col => !cabecalho.includes(col));
  if (colunasAusentes.length > 0) {
    avisar('Colunas ausentes na planilha: ' + colunasAusentes.join(', '));
    return;
  }

  const linhas = dados.slice(1);
  if (linhas.length === 0) {
    avisar('Não há clientes na planilha.');
    return;
  }

  // linhas em branco no meio da lista deslocariam os scores para os clientes errados
  const linhaEmBranco = linhas.findIndex(linha => COLUNAS_OBRIGATORIAS.every(col => linha[cabecalho.indexOf(col)] === ''));
  if (linhaEmBranco >= 0) {
    avisar(`A linha ${linhaEmBranco + 2} está em branco. Remova as linhas em branco e tente novamente.`);
    return;
  }

  // colunas de resultado: reaproveita as de execuções anteriores ou cria no final
  const colScore = obterColuna(sheet, cabecalho, COLUNA_SCORE);
  const colRanking = obterColuna(sheet, cabecalho, COLUNA_RANKING);

  // converte cada linha em um registro { coluna: valor } com as colunas obrigatórias
  const registros = linhas.map(linha => {
    const registro = {};
    COLUNAS_OBRIGATORIAS.forEach(col => registro[col] = linha[cabecalho.indexOf(col)]);
    return registro;
  });

  notificar('Conectando à API...');
  if (!acordarApi()) {
    avisar('A API não respondeu. Tente novamente em alguns minutos.');
    return;
  }

  // envia os clientes em lotes e guarda o score de cada um, na mesma ordem das linhas
  const scores = [];
  for (let inicio = 0; inicio < registros.length; inicio += TAMANHO_LOTE) {
    const lote = registros.slice(inicio, inicio + TAMANHO_LOTE);
    notificar(`Calculando o score dos clientes ${inicio + 1} a ${inicio + lote.length} de ${registros.length}...`);

    const resposta = UrlFetchApp.fetch(API_URL + '/allinsurance/predict', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(lote),
      muteHttpExceptions: true
    });

    if (resposta.getResponseCode() !== 200) {
      avisar(`Erro na API ao calcular os clientes ${inicio + 1} a ${inicio + lote.length}: ` + resposta.getContentText());
      return;
    }

    JSON.parse(resposta.getContentText()).forEach(cliente => scores.push([cliente.score]));
  }

  // escreve o score de cada cliente e limpa os destaques de execuções anteriores
  const totalClientes = scores.length;
  const dadosRange = sheet.getRange(2, 1, totalClientes, sheet.getLastColumn());
  dadosRange.setBackground(null);
  sheet.getRange(2, colScore, totalClientes, 1).setValues(scores).setNumberFormat('0.0%');

  // ordena os clientes do maior para o menor score
  dadosRange.sort({ column: colScore, ascending: false });

  // escreve a posição de cada cliente no ranking
  const ranking = scores.map((_, i) => [i + 1]);
  sheet.getRange(2, colRanking, totalClientes, 1).setValues(ranking);

  notificar(`${totalClientes} clientes ordenados pela propensão de compra.`, 5);
}


function destacarClientes() {
  const ui = SpreadsheetApp.getUi();
  const sheet = SpreadsheetApp.getActiveSheet();
  const totalClientes = sheet.getLastRow() - 1;

  const cabecalho = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(String);
  const colRanking = cabecalho.indexOf(COLUNA_RANKING) + 1;
  if (colRanking === 0 || totalClientes <= 0) {
    avisar('Calcule a propensão de compra antes de destacar os clientes.');
    return;
  }

  const resposta = ui.prompt('Destacar clientes para ligação', 'Quantas ligações o time de vendas consegue fazer?', ui.ButtonSet.OK_CANCEL);
  if (resposta.getSelectedButton() !== ui.Button.OK) return;

  // aceita o número com separador de milhar, como 20.000
  const ligacoes = parseInt(resposta.getResponseText().replace(/\D/g, ''), 10);
  if (isNaN(ligacoes) || ligacoes <= 0) {
    avisar('Informe um número de ligações válido.');
    return;
  }

  // destaca os clientes pelo ranking, mesmo que a planilha tenha sido reordenada
  const rankings = sheet.getRange(2, colRanking, totalClientes, 1).getValues();
  const cores = rankings.map(([posicao]) => Array(sheet.getLastColumn()).fill(posicao !== '' && posicao <= ligacoes ? COR_DESTAQUE : null));
  sheet.getRange(2, 1, totalClientes, sheet.getLastColumn()).setBackgrounds(cores);

  const destacados = rankings.filter(([posicao]) => posicao !== '' && posicao <= ligacoes).length;
  notificar(`${destacados} clientes destacados para ligação.`, 5);
}


// no plano Eco do Heroku, a API "dorme" após 30 minutos sem uso e leva alguns
// segundos para acordar: tenta algumas vezes antes de desistir
function acordarApi() {
  for (let tentativa = 1; tentativa <= 6; tentativa++) {
    try {
      const resposta = UrlFetchApp.fetch(API_URL, { muteHttpExceptions: true });
      if (resposta.getResponseCode() === 200) return true;
    } catch (e) {
      // a requisição pode expirar enquanto a API está acordando
    }
    Utilities.sleep(10000);
  }
  return false;
}


// devolve o número da coluna com o nome informado, criando-a no final da planilha se não existir
function obterColuna(sheet, cabecalho, nome) {
  const indice = cabecalho.indexOf(nome);
  if (indice >= 0) return indice + 1;

  const novaColuna = sheet.getLastColumn() + 1;
  sheet.getRange(1, novaColuna).setValue(nome);
  cabecalho.push(nome);
  return novaColuna;
}


// mensagem de progresso no canto da planilha (segundos = -1 mantém a mensagem até a próxima)
function notificar(mensagem, segundos = -1) {
  SpreadsheetApp.getActive().toast(mensagem, 'All Insurance', segundos);
}


// remove a mensagem de progresso e mostra um alerta
function avisar(mensagem) {
  SpreadsheetApp.getActive().toast('', 'All Insurance', 1);
  SpreadsheetApp.getUi().alert(mensagem);
}
