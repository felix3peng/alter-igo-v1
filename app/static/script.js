$(document).ready(function() {
    $('.inputform').on('submit', function(e) {
        $.ajax({
            url: '/process',
            type: 'POST',
            cache: false,
            dataType: 'json',
            data : {cmd : $('#cmd').val()}
        }).done(function(response) {
            $('.content').append('<div class="command"><p>'+response.outputs['txtcmd']+'</p></div>');
            //$('.content').append('<div class="code"><code>'+response.outputs['rawcode']+'</code></div>');
            //$('.content').append('<div class="output"><pre>'+response.outputs['output']+'</pre></div>');
            $('#cmd').value = "";
        });
        e.preventDefault();
    });
});