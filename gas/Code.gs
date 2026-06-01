const SPREADSHEET_ID = "1oNTdGF24ijbKK3HQbOoA5XA-7g_tBIz-2ILe6O-8pjA";
const SHEET_INDEX = 0;

function getTargetSheet() {
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheet = spreadsheet.getSheets()[SHEET_INDEX];

  if (!sheet) {
    throw new Error("Target sheet not found.");
  }

  return sheet;
}

function ensureHeader(sheet) {
  if (sheet.getLastRow() > 0) {
    return;
  }

  sheet.appendRow(["date", "style", "q1", "q2", "time", "q4"]);
}

function doPost(e) {
  try {
    if (!e || !e.parameter) {
      return ContentService
        .createTextOutput("Error: no form data received")
        .setMimeType(ContentService.MimeType.TEXT);
    }

    const sheet = getTargetSheet();
    ensureHeader(sheet);

    const row = [
      e.parameter.date || "",
      e.parameter.style || "",
      e.parameter.q1 || "",
      e.parameter.q2 || "",
      e.parameter.time || "",
      e.parameter.q4 || ""
    ];

    sheet.appendRow(row);

    return ContentService
      .createTextOutput("Success")
      .setMimeType(ContentService.MimeType.TEXT);
  } catch (error) {
    return ContentService
      .createTextOutput(`Error: ${error.message}`)
      .setMimeType(ContentService.MimeType.TEXT);
  }
}
